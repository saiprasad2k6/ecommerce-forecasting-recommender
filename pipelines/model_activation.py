"""Manual activation after the run's DVC metadata is committed and reviewed."""
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shlex
import sys

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

ROOT = Path(os.environ.get("OLIST_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DEFAULT_PYTHON = ROOT / ".venv/bin/python"
PYTHON = os.environ.get("OLIST_PYTHON", str(DEFAULT_PYTHON) if DEFAULT_PYTHON.exists() else sys.executable)
with DAG(
    dag_id="olist_model_activation", schedule=None, catchup=False,
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc), max_active_runs=1, max_active_tasks=1,
    params={"training_run_id": Param(type="string", minLength=1)},
    default_args={"owner": "ecommerce", "retries": 1, "retry_delay": timedelta(minutes=5)},
    doc_md="Commit the uploaded training run metadata, review its gates, then select its training_run_id. Development runs cannot be activated.",
) as dag:
    activate = BashOperator(
        task_id="activate_models", cwd=str(ROOT),
        bash_command=shlex.join([PYTHON, str(ROOT / "scripts/models/activate_models.py")]) + ' --pipeline-run-id "$TRAINING_RUN_ID"',
        env={"TRAINING_RUN_ID": "{{ params.training_run_id }}"}, append_env=True,
        do_xcom_push=False, skip_on_exit_code=None,
    )
