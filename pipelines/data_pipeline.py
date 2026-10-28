"""Airflow 3 DAG: download Olist, prepare features and evaluate baselines.

The project checkout and Python environment must be available to every worker.
Override OLIST_PROJECT_ROOT / OLIST_PYTHON when deploying outside this checkout.
No dataset access or model evaluation happens when the scheduler parses this file.
"""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shlex
import sys

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param


ROOT = Path(os.environ.get("OLIST_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
PROJECT_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = os.environ.get("OLIST_PYTHON", str(PROJECT_PYTHON) if PROJECT_PYTHON.is_file() else sys.executable)
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
CURATED = ROOT / "data/curated"
WAREHOUSE = ROOT / "data/warehouse/olist.duckdb"
FEATURES = ROOT / "data/features"
BASELINES = ROOT / "data/baselines"
TRACKING_URI = f"sqlite:///{BASELINES / 'mlflow.db'}"


def script_task(task_id: str, script: str, *arguments: str, env: dict | None = None) -> BashOperator:
    """Run an existing CLI; inherit credentials and expose its Kaggle executable.

    The arguments are fixed paths/flags. User-supplied Params are passed through
    quoted shell environment variables rather than interpolated into shell code.
    """
    command = shlex.join([PYTHON, str(ROOT / "scripts" / script)])
    if arguments:
        command += " " + " ".join(arguments)
    return BashOperator(
        task_id=task_id,
        bash_command=f'export PATH={shlex.quote(str(Path(PYTHON).parent))}:"$PATH"\nexec {command}',
        cwd=str(ROOT),
        env={"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "MLFLOW_DISABLE_AGENT_HINT": "1", **(env or {})},
        append_env=True,
        do_xcom_push=False,
        skip_on_exit_code=None,
    )


with DAG(
    dag_id="olist_end_to_end_pipeline",
    description="Download, validate, clean, curate, build features and evaluate Olist baselines",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    default_args={"owner": "ecommerce", "retries": 1,
                  "retry_delay": timedelta(minutes=5),
                  "execution_timeout": timedelta(hours=6)},
    tags=["olist", "features", "baselines"],
    params={
        "validation_days": Param(30, type="integer", minimum=1),
        "test_days": Param(30, type="integer", minimum=1),
        "recommendation_customer_id": Param("new-customer", type="string", minLength=1,
            description="customer_unique_id; the default exercises the new-customer popularity fallback"),
        "recommendation_state": Param("SP", type="string", pattern="^[A-Z]{2}$"),
        "recommendation_as_of": Param("2018-08-01", type="string", format="date",
            description="Historical recommendation feature cutoff, within the Olist data period"),
        "top_k": Param(10, type="integer", minimum=1),
    },
    doc_md="""
    ## Olist end-to-end pipeline
    Trigger manually after configuring Kaggle credentials on the worker. Existing
    complete raw downloads are reused. No task deletes or resets repository data.

    Validation and cleaning must succeed before warehouse creation. Both feature
    builders run before either baseline evaluation. Recommendation features are a
    single-customer snapshot configured in Params; recommendation baseline evaluation
    builds its own historical customer cohorts from the warehouse (July/August 2018).

    All tasks use shared local paths and execute serially, including MLflow writers.
    Workers need the same project checkout/storage and installed dependencies.
    """,
) as dag:
    download = script_task("download_olist_data", "data/download_olist_data.py",
                           "--output-dir", shlex.quote(str(RAW)))
    validate = script_task("validate_olist_data", "data/validate_olist_data.py",
                           "--input-dir", shlex.quote(str(RAW)),
                           "--schema", shlex.quote(str(ROOT / "configs/schema.yaml")))
    clean = script_task("clean_olist_data", "data/clean_olist_data.py",
                        "--input-dir", shlex.quote(str(RAW)),
                        "--output-dir", shlex.quote(str(PROCESSED)))
    curate = script_task("prepare_curated_data", "data/prepare_curated_data.py",
                         "--input-dir", shlex.quote(str(PROCESSED)),
                         "--output-dir", shlex.quote(str(CURATED)),
                         "--warehouse", shlex.quote(str(WAREHOUSE)))
    forecasting_features = script_task(
        "build_forecasting_features", "features/build_forecasting_features.py",
        "--input-dir", shlex.quote(str(CURATED)),
        "--output-dir", shlex.quote(str(FEATURES / "forecasting")),
        "--validation-days", '"$OLIST_VALIDATION_DAYS"',
        "--test-days", '"$OLIST_TEST_DAYS"',
        env={"OLIST_VALIDATION_DAYS": "{{ params.validation_days }}",
             "OLIST_TEST_DAYS": "{{ params.test_days }}"},
    )
    recommendation_features = script_task(
        "build_recommendation_features", "features/build_recommendation_features.py",
        "--warehouse", shlex.quote(str(WAREHOUSE)),
        "--output-dir", shlex.quote(str(FEATURES / "recommendations")),
        "--customer-id", '"$OLIST_CUSTOMER_ID"',
        "--state", '"$OLIST_STATE"', "--as-of", '"$OLIST_AS_OF"',
        "--top-k", '"$OLIST_TOP_K"',
        env={"OLIST_CUSTOMER_ID": "{{ params.recommendation_customer_id }}",
             "OLIST_STATE": "{{ params.recommendation_state }}",
             "OLIST_AS_OF": "{{ params.recommendation_as_of }}",
             "OLIST_TOP_K": "{{ params.top_k }}"},
    )
    forecasting_baselines = script_task(
        "run_forecasting_baselines", "models/baseline/forecasting_baselines.py",
        "--features-dir", shlex.quote(str(FEATURES / "forecasting")),
        "--output-dir", shlex.quote(str(BASELINES / "forecasting")),
        "--tracking-uri", shlex.quote(TRACKING_URI),
        "--experiment-name", "ecommerce-demand-forecasting",
    )
    recommendation_baselines = script_task(
        "run_recommendation_baselines", "models/baseline/recommendation_baselines.py",
        "--warehouse", shlex.quote(str(WAREHOUSE)),
        "--output-dir", shlex.quote(str(BASELINES / "recommendations")),
        "--tracking-uri", shlex.quote(TRACKING_URI),
        "--experiment-name", "ecommerce-recommendations",
        "--top-k", '"$OLIST_TOP_K"',
        env={"OLIST_TOP_K": "{{ params.top_k }}"},
    )

    (download >> validate >> clean >> curate >> forecasting_features
     >> recommendation_features >> forecasting_baselines >> recommendation_baselines)
