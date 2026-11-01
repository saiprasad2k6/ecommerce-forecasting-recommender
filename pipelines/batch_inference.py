"""Manual batch inference using only models activated by comparison."""
from datetime import datetime, timedelta, timezone
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
PYTHON = os.environ.get("OLIST_PYTHON", str(PROJECT_PYTHON) if PROJECT_PYTHON.is_file() else sys.executable)

with DAG(
    dag_id="olist_batch_inference",
    description="Predict using the approved active forecasting and recommendation models",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc), schedule=None, catchup=False,
    max_active_runs=1, max_active_tasks=1,
    default_args={"owner": "ecommerce", "retries": 1, "retry_delay": timedelta(minutes=5)},
    params={
        "as_of": Param(SETTINGS["inference"]["as_of"], type="string", format="date"),
        "forecast_features": Param(SETTINGS["inference"]["forecast_features"], type="string", minLength=1),
        "warehouse": Param(SETTINGS["inference"]["warehouse"], type="string", minLength=1),
        "models_dir": Param(SETTINGS["inference"]["models_dir"], type="string", minLength=1),
        "output_dir": Param(SETTINGS["inference"]["output_dir"], type="string", minLength=1),
    },
    tags=["olist", "inference"],
    doc_md="""
    Run training/comparison first to publish serving-approved model bundles.
    Defaults replay historical inputs; set as_of and input paths for each batch.
    Forecasting predicts the day after its feature date. Recommendations use only
    history before as_of. Neither task selects or trains models.
    """,
) as dag:
    tasks = []
    for task, input_parameter in [("forecasting", "forecast_features"), ("recommendation", "warehouse")]:
        tasks.append(BashOperator(
            task_id=f"predict_{task}", cwd=str(ROOT),
            bash_command=(shlex.join([PYTHON, str(ROOT / "scripts/inference/run_batch.py"), "--task", task])
                          + ' --as-of "$AS_OF" --input-path "$INPUT_PATH" --models-dir "$MODELS_DIR"'
                          + ' --output-dir "$OUTPUT_DIR" --pipeline-run-id "$PIPELINE_RUN_ID"'),
            env={"AS_OF": "{{ params.as_of }}", "INPUT_PATH": "{{ params." + input_parameter + " }}",
                 "MODELS_DIR": "{{ params.models_dir }}", "OUTPUT_DIR": "{{ params.output_dir }}",
                 "PIPELINE_RUN_ID": "{{ run_id }}", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
            append_env=True, do_xcom_push=False, skip_on_exit_code=None,
        ))
    tasks[0] >> tasks[1]
