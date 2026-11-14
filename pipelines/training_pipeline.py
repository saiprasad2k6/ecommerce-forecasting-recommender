"""Manual model training DAG; run olist_data_preparation first."""

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
PROJECT_PYTHON = ROOT / ".venv/bin/python"
PYTHON = os.environ.get("OLIST_PYTHON", str(PROJECT_PYTHON) if PROJECT_PYTHON.is_file() else sys.executable)
MODEL_JOBS = json.loads((ROOT / "configs/production_models.json").read_text())


def training_task(task_id, script, arguments="", *, env=None,
                  run_id=True, trigger_rule="all_success"):
    """Run a project CLI; dynamic values use quoted environment variables."""
    command = shlex.join([PYTHON, str(ROOT / "scripts" / script)])
    if run_id:
        command += ' --pipeline-run-id "$PIPELINE_RUN_ID"'
    return BashOperator(
        task_id=task_id,
        bash_command=f'export PATH={shlex.quote(str(Path(PYTHON).parent))}:"$PATH"\nexec {command} {arguments}',
        cwd=str(ROOT),
        env={"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "PIPELINE_RUN_ID": "{{ run_id }}", **(env or {})},
        append_env=True, do_xcom_push=False, skip_on_exit_code=None,
        trigger_rule=trigger_rule,
    )


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
        **{name: Param(REPLAY[name], type="string", format="date") for name in
           ["training_start", "validation_start", "test_start", "monitoring_start"]},
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
        training_task("select_prepared_version", "versioning/snapshot.py",
                      'begin-training --prepared-version "$PREPARED_VERSION" '
                      '{{ "" if params.release else "--development" }}',
                      env={"OLIST_PARAMETERS": "{{ params | tojson }}",
                           "PREPARED_VERSION": "{{ params.prepared_version }}"}),
        *[
            training_task(f"evaluate_{task}_baselines", "models/run_job.py",
                          shlex.join(["--task", task, "--model", "baselines"]))
            for task in ("forecasting", "recommendation")
        ],
        *[
            training_task(f"train_{task}_{model}", "models/run_job.py",
                          shlex.join(["--task", task, "--model", model]))
            for task, models in MODEL_JOBS.items() for model in models
        ],
        training_task("compare_models", "models/comparison/compare_models.py"),
        training_task("preserve_run_snapshot", "versioning/snapshot.py", "artifacts"),
        training_task("backup_mlflow", "versioning/snapshot.py", "backup-mlflow", run_id=False),
    ]
    release = training_task("release_workspace", "versioning/snapshot.py",
                            "release-workspace --kind train", trigger_rule="all_done")
    watcher = BashOperator(task_id="workflow_failed", bash_command="exit 1",
                           trigger_rule="one_failed", retries=0)
    chain(*steps)
    # Always release ownership; the watcher preserves failures in the DAG result.
    for task in steps:
        task >> release
        task >> watcher
    release >> watcher
