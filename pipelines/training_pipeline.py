"""Airflow 3 DAG: train, evaluate and preserve models from a prepared version.

Run olist_data_preparation and commit its snapshot metadata before training.
Override OLIST_PROJECT_ROOT / OLIST_PYTHON when deploying outside this checkout.
No training or dataset access happens when the scheduler parses this file.
"""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shlex
import sys

import yaml

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param, chain


ROOT = Path(os.environ.get("OLIST_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
SETTINGS = yaml.safe_load((ROOT / "configs/settings.yaml").read_text())
REPLAY = SETTINGS["replay"]
PROJECT_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = os.environ.get("OLIST_PYTHON", str(PROJECT_PYTHON) if PROJECT_PYTHON.is_file() else sys.executable)
SPLIT_DATES = ("training_start", "validation_start", "test_start", "monitoring_start")
MODEL_JOBS = json.loads((ROOT / "configs/production_models.json").read_text())


def script_task(task_id: str, script: str, *arguments: str,
                parameters: dict[str, str] | None = None,
                env: dict[str, str] | None = None, run_id: bool = True,
                trigger_rule: str = "all_success") -> BashOperator:
    """Quote literal arguments and pass trigger parameters through shell variables.

    ``parameters`` maps CLI flag names to Airflow parameter names.
    Model CLIs enforce workspace ownership using the training run manifest.
    """
    command = shlex.join([PYTHON, str(ROOT / "scripts" / script), *arguments])
    environment = {"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "PIPELINE_RUN_ID": "{{ run_id }}", **(env or {})}
    for flag, parameter in (parameters or {}).items():
        variable = "OLIST_" + parameter.upper()
        environment[variable] = "{{ params." + parameter + " }}"
        command += f' --{flag} "${variable}"'
    if run_id:
        command += ' --pipeline-run-id "$PIPELINE_RUN_ID"'
    if script == "versioning/snapshot.py" and arguments[0] == "begin-training":
        command += ' {{ "" if params.release else "--development" }}'
    return BashOperator(
        task_id=task_id,
        bash_command=f'export PATH={shlex.quote(str(Path(PYTHON).parent))}:"$PATH"\nexec {command}',
        cwd=str(ROOT), env=environment, append_env=True,
        do_xcom_push=False, skip_on_exit_code=None, trigger_rule=trigger_rule,
    )


def snapshot_task(task_id: str, action: str, *arguments: str, **kwargs) -> BashOperator:
    """Manage training checkpoints; MLflow backup does not take a run ID."""
    return script_task(task_id, "versioning/snapshot.py", action, *arguments,
                       run_id=action != "backup-mlflow", **kwargs)


with DAG(
    dag_id="olist_model_training",
    description="Train from a published prepared version, compare and preserve release artifacts",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    default_args={
        "owner": "ecommerce",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=6),
    },
    tags=["olist", "models", "training"],
    params={
        "prepared_version": Param(type="string", pattern="^[0-9a-f]{64}$", description="Published prepared version ID; no latest-data fallback"),
        "release": Param(SETTINGS["release"]["require_clean_git"], type="boolean"),
        **{name: Param(REPLAY[name], type="string", format="date") for name in SPLIT_DATES},
        "max_train_rows": Param(SETTINGS["training"]["max_train_rows"], type="integer", minimum=2),
        "forecast_estimators": Param(SETTINGS["training"]["forecast_estimators"], type="integer", minimum=1),
        "ranking_estimators": Param(SETTINGS["training"]["ranking_estimators"], type="integer", minimum=1),
        "seed": Param(SETTINGS["training"]["seed"], type="integer", minimum=0),
        "candidate_limit": Param(SETTINGS["recommendation"]["candidate_limit"], type="integer", minimum=10),
        "top_k": Param(SETTINGS["recommendation"]["top_k"], type="integer", minimum=1),
    },
    doc_md="""
    Prepare data and commit its snapshot metadata before training. Jobs use the
    selected version and frozen parameters; use a new run ID to retrain.
    Compare and preserve models, then commit run metadata before activating them
    through the separate activation DAG.
    """,
) as dag:
    steps = [
        snapshot_task("select_prepared_version", "begin-training",
                      parameters={"prepared-version": "prepared_version"},
                      env={"OLIST_PARAMETERS": "{{ params | tojson }}"}),
        *[
            script_task(f"evaluate_{task}_baselines", "models/run_job.py",
                        "--task", task, "--model", "baselines")
            for task in ("forecasting", "recommendation")
        ],
        *[
            script_task(f"train_{task}_{model}", "models/run_job.py",
                        "--task", task, "--model", model)
            for task, models in MODEL_JOBS.items() for model in models
        ],
        script_task("compare_models", "models/comparison/compare_models.py"),
        snapshot_task("preserve_run_snapshot", "artifacts"),
        snapshot_task("backup_mlflow", "backup-mlflow"),
    ]
    release = snapshot_task("release_workspace", "release-workspace", "--kind", "train",
                            trigger_rule="all_done")
    watcher = BashOperator(task_id="workflow_failed", bash_command="exit 1",
                           trigger_rule="one_failed", retries=0)
    chain(*steps)
    # Always release ownership; the watcher preserves failures in the DAG result.
    for task in steps:
        task >> release
        task >> watcher
    release >> watcher
