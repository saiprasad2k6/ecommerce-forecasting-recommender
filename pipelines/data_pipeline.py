"""Airflow 3 DAG: download Olist and prepare data and features.

The project checkout and Python environment must be available to every worker.
Override OLIST_PROJECT_ROOT / OLIST_PYTHON when deploying outside this checkout.
No dataset access or feature preparation happens when the scheduler parses this file.
"""

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
PROJECT_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = os.environ.get("OLIST_PYTHON", str(PROJECT_PYTHON) if PROJECT_PYTHON.is_file() else sys.executable)
RAW = ROOT / SETTINGS["data"]["raw_dir"]
PROCESSED = ROOT / SETTINGS["data"]["processed_dir"]
CURATED = ROOT / SETTINGS["data"]["curated_dir"]
WAREHOUSE = ROOT / SETTINGS["data"]["warehouse"]
FORECASTING_FEATURES = ROOT / SETTINGS["data"]["forecasting_features"]
RECOMMENDATION_FEATURES = ROOT / SETTINGS["data"]["recommendation_features"]


def script_task(task_id: str, script: str, *arguments: str, env: dict | None = None, managed=True) -> BashOperator:
    """Run an existing CLI; inherit credentials and expose its Kaggle executable.

    The arguments are fixed paths/flags. User-supplied Params are passed through
    quoted shell environment variables rather than interpolated into shell code.
    """
    command = shlex.join([PYTHON, str(ROOT / "scripts" / script)])
    if arguments:
        command += " " + " ".join(arguments)
    if managed:
        prefix = shlex.join([PYTHON, str(ROOT / "scripts/versioning/snapshot.py"), "execute-preparation"])
        command = prefix + ' --pipeline-run-id "$PREP_RUN_ID" --stage ' + shlex.quote(task_id) + ' -- ' + command
    return BashOperator(
        task_id=task_id,
        bash_command=f'export PATH={shlex.quote(str(Path(PYTHON).parent))}:"$PATH"\nexec {command}',
        cwd=str(ROOT),
        env={"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "MLFLOW_DISABLE_AGENT_HINT": "1", "PREP_RUN_ID": "{{ run_id }}", **(env or {})},
        append_env=True,
        do_xcom_push=False,
        skip_on_exit_code=None,
    )


with DAG(
    dag_id="olist_data_preparation",
    description="Download, validate, clean and curate Olist data, then build features",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    default_args={"owner": "ecommerce", "retries": 1,
                  "retry_delay": timedelta(minutes=5),
                  "execution_timeout": timedelta(hours=6)},
    tags=["olist", "data-preparation", "features"],
    params={
        "release": Param(SETTINGS["release"]["require_clean_git"], type="boolean"),
        **{name: Param(REPLAY[name], type="string", format="date") for name in
           ["validation_start", "test_start", "monitoring_start", "monitoring_end"]},
        "recommendation_customer_id": Param("new-customer", type="string", minLength=1,
            description="customer_unique_id; the default exercises the new-customer popularity fallback"),
        "recommendation_state": Param("SP", type="string", pattern="^[A-Z]{2}$"),
        "recommendation_as_of": Param(REPLAY["monitoring_start"], type="string", format="date",
            description="Historical recommendation feature cutoff, within the Olist data period"),
        "top_k": Param(SETTINGS["recommendation"]["top_k"], type="integer", minimum=1),
    },
    doc_md="""
    ## Olist data preparation
    Trigger manually after configuring Kaggle credentials on the worker. Existing
    complete raw downloads are reused. No task deletes or resets repository data.

    Validation and cleaning must succeed before warehouse creation. Recommendation
    features are a single-customer snapshot configured in Params. After this DAG
    succeeds, trigger olist_model_training to evaluate baselines, train candidates
    and publish the selected models.

    All tasks use shared local paths and execute serially.
    Workers need the same project checkout/storage and installed dependencies.
    """,
) as dag:
    begin = script_task("begin_preparation", "versioning/snapshot.py",
        'begin-preparation --pipeline-run-id "$PREP_RUN_ID" {{ "" if params.release else "--development" }}',
        env={"OLIST_PARAMETERS": "{{ params | tojson }}"}, managed=False)
    download = script_task("download_olist_data", "data_processing/download_olist_data.py",
                           "--output-dir", shlex.quote(str(RAW)),
                           "--dataset", shlex.quote(SETTINGS["data"]["dataset"]))
    validate = script_task("validate_olist_data", "data_processing/validate_olist_data.py",
                           "--input-dir", shlex.quote(str(RAW)),
                           "--schema", shlex.quote(str(ROOT / "configs/schema.yaml")))
    raw = script_task("publish_raw_version", "versioning/snapshot.py",
        'raw --pipeline-run-id "$PREP_RUN_ID"', managed=False)
    clean = script_task("clean_olist_data", "data_processing/clean_olist_data.py",
                        "--input-dir", shlex.quote(str(RAW)),
                        "--output-dir", shlex.quote(str(PROCESSED)))
    curate = script_task("prepare_curated_data", "data_processing/prepare_curated_data.py",
                         "--input-dir", shlex.quote(str(PROCESSED)),
                         "--output-dir", shlex.quote(str(CURATED)),
                         "--warehouse", shlex.quote(str(WAREHOUSE)))
    forecasting_features = script_task(
        "build_forecasting_features", "features/build_forecasting_features.py",
        "--input-dir", shlex.quote(str(CURATED)),
        "--output-dir", shlex.quote(str(FORECASTING_FEATURES)),
        "--validation-start", '"$VALIDATION_START"',
        "--test-start", '"$TEST_START"',
        "--monitoring-start", '"$MONITORING_START"',
        "--monitoring-end", '"$MONITORING_END"',
        env={name.upper(): "{{ params." + name + " }}" for name in
             ["validation_start", "test_start", "monitoring_start", "monitoring_end"]},
    )
    recommendation_features = script_task(
        "build_recommendation_features", "features/build_recommendation_features.py",
        "--warehouse", shlex.quote(str(WAREHOUSE)),
        "--output-dir", shlex.quote(str(RECOMMENDATION_FEATURES)),
        "--customer-id", '"$OLIST_CUSTOMER_ID"',
        "--state", '"$OLIST_STATE"', "--as-of", '"$OLIST_AS_OF"',
        "--top-k", '"$OLIST_TOP_K"',
        "--candidate-limit", '"$OLIST_CANDIDATES"',
        env={"OLIST_CUSTOMER_ID": "{{ params.recommendation_customer_id }}",
             "OLIST_STATE": "{{ params.recommendation_state }}",
             "OLIST_AS_OF": "{{ params.recommendation_as_of }}",
             "OLIST_TOP_K": "{{ params.top_k }}",
             "OLIST_CANDIDATES": str(SETTINGS["recommendation"]["candidate_limit"])},
    )
    publish = script_task("publish_prepared_version", "versioning/snapshot.py",
        'prepared --pipeline-run-id "$PREP_RUN_ID"', managed=False)
    release = script_task("release_workspace", "versioning/snapshot.py",
        'release-workspace --kind prepare --pipeline-run-id "$PREP_RUN_ID"', managed=False)
    release.trigger_rule = "all_done"
    watcher = BashOperator(task_id="workflow_failed", bash_command="exit 1", trigger_rule="one_failed", retries=0)
    (begin >> download >> validate >> raw >> clean >> curate >> forecasting_features
     >> recommendation_features >> publish)
    business = [begin, download, validate, raw, clean, curate, forecasting_features, recommendation_features, publish]
    for task in business:
        task >> release
        task >> watcher
    release >> watcher
