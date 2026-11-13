"""Manual model training DAG; run olist_data_preparation first."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shlex
import sys

import yaml

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param


ROOT = Path(os.environ.get("OLIST_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
SETTINGS = yaml.safe_load((ROOT / "configs/settings.yaml").read_text())
REPLAY = SETTINGS["replay"]
PROJECT_PYTHON = ROOT / ".venv/bin/python"
DEFAULT_PYTHON = str(PROJECT_PYTHON) if PROJECT_PYTHON.is_file() else sys.executable
PYTHON = os.environ.get("OLIST_PYTHON", DEFAULT_PYTHON)
MODEL_JOBS = json.loads((ROOT / "configs/production_models.json").read_text())


def training_task(task_id, script, arguments, env):
    """Run a CLI wrapper with parameters passed through quoted environment values."""
    command = shlex.join([PYTHON, str(ROOT / "scripts/models" / script)])
    python_bin = shlex.quote(str(Path(PYTHON).parent))
    return BashOperator(
        task_id=task_id,
        bash_command=f'export PATH={python_bin}:"$PATH"\nexec {command} {arguments}',
        cwd=str(ROOT),
        env={"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "PIPELINE_RUN_ID": "{{ run_id }}", **env},
        append_env=True,
        do_xcom_push=False,
        skip_on_exit_code=None,
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
    Run olist_data_preparation, then commit its snapshots/ metadata before
    training. All jobs use that version and the resolved parameters in its manifest.
    Comparison only prepares selected bundles. Upload the complete run and back up
    MLflow before a separately triggered activation release. Rejected candidates retain existing active models;
    a first deployment can activate the selected baseline fallback.
    Commit snapshots/ metadata through the normal Git review/release workflow.
    Use a new run ID to retrain; retries reuse the frozen inputs.
    """,
) as dag:
    snapshot = training_task(
        "select_prepared_version", "../versioning/snapshot.py",
        'begin-training --pipeline-run-id "$PIPELINE_RUN_ID" --prepared-version "$PREPARED_VERSION" {{ "" if params.release else "--development" }}',
        env={"OLIST_PARAMETERS": "{{ params | tojson }}", "PREPARED_VERSION": "{{ params.prepared_version }}"},
    )
    previous = snapshot
    for task in ["forecasting", "recommendation"]:
        baseline = training_task(
            f"evaluate_{task}_baselines", "run_job.py",
            f'--pipeline-run-id "$PIPELINE_RUN_ID" --task {task} --model baselines', env={},
        )
        previous >> baseline
        previous = baseline
    for task, models in MODEL_JOBS.items():
        for model_name in models:
            training = training_task(
                f"train_{task}_{model_name}", "run_job.py",
                '--pipeline-run-id "$PIPELINE_RUN_ID" --task "$MODEL_TASK" --model "$MODEL_NAME"',
                env={"MODEL_TASK": task, "MODEL_NAME": model_name},
            )
            previous >> training
            previous = training
    comparison = training_task(
        "compare_models", "comparison/compare_models.py",
        '--pipeline-run-id "$PIPELINE_RUN_ID"', env={},
    )
    preserve = training_task(
        "preserve_run_snapshot", "../versioning/snapshot.py",
        'artifacts --pipeline-run-id "$PIPELINE_RUN_ID"', env={},
    )
    backup = training_task(
        "backup_mlflow", "../versioning/snapshot.py", "backup-mlflow", env={},
    )
    release = training_task(
        "release_workspace", "../versioning/snapshot.py",
        'release-workspace --kind train --pipeline-run-id "$PIPELINE_RUN_ID"', env={},
    )
    release.trigger_rule = "all_done"
    watcher = BashOperator(task_id="workflow_failed", bash_command="exit 1", trigger_rule="one_failed", retries=0)
    previous >> comparison >> preserve >> backup
    business = [task for task in dag.tasks if task not in [release, watcher]]
    for task in business:
        task >> release
        task >> watcher
    release >> watcher
