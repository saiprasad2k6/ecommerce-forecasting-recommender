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
from airflow.sdk import DAG, Param, chain


ROOT = Path(os.environ.get("OLIST_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
SETTINGS = yaml.safe_load((ROOT / "configs/settings.yaml").read_text())
REPLAY = SETTINGS["replay"]
PROJECT_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = os.environ.get("OLIST_PYTHON", str(PROJECT_PYTHON) if PROJECT_PYTHON.is_file() else sys.executable)
SPLIT_DATES = ("validation_start", "test_start", "monitoring_start", "monitoring_end")


def data_path(name: str) -> str:
    return str(ROOT / SETTINGS["data"][name])


def script_task(task_id: str, script: str, *arguments: str,
                parameters: dict[str, str] | None = None,
                env: dict[str, str] | None = None, managed: bool = True,
                trigger_rule: str = "all_success") -> BashOperator:
    """Quote literal arguments and pass trigger parameters through shell variables.

    Managed tasks run under the preparation workspace ownership guard.
    ``parameters`` maps CLI flag names to Airflow parameter names.
    """
    command = shlex.join([PYTHON, str(ROOT / "scripts" / script), *arguments])
    environment = {"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "PREP_RUN_ID": "{{ run_id }}", **(env or {})}
    for flag, parameter in (parameters or {}).items():
        variable = "OLIST_" + parameter.upper()
        environment[variable] = "{{ params." + parameter + " }}"
        command += f' --{flag} "${variable}"'
    if script == "versioning/snapshot.py":
        command += ' --pipeline-run-id "$PREP_RUN_ID"'
        if arguments[0] == "begin-preparation":
            command += ' {{ "" if params.release else "--development" }}'
    if managed:
        prefix = shlex.join([PYTHON, str(ROOT / "scripts/versioning/snapshot.py"), "execute-preparation"])
        command = prefix + ' --pipeline-run-id "$PREP_RUN_ID" --stage ' + shlex.quote(task_id) + ' -- ' + command
    return BashOperator(
        task_id=task_id,
        bash_command=f'export PATH={shlex.quote(str(Path(PYTHON).parent))}:"$PATH"\nexec {command}',
        cwd=str(ROOT), env=environment, append_env=True,
        do_xcom_push=False, skip_on_exit_code=None, trigger_rule=trigger_rule,
    )


def snapshot_task(task_id: str, action: str, *arguments: str, **kwargs) -> BashOperator:
    """Publish or manage a preparation checkpoint outside the stage guard."""
    return script_task(task_id, "versioning/snapshot.py", action, *arguments,
                       managed=False, **kwargs)


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
        **{name: Param(REPLAY[name], type="string", format="date") for name in SPLIT_DATES},
        "recommendation_customer_id": Param("new-customer", type="string", minLength=1,
            description="customer_unique_id; the default exercises the new-customer popularity fallback"),
        "recommendation_state": Param("SP", type="string", pattern="^[A-Z]{2}$"),
        "recommendation_as_of": Param(REPLAY["monitoring_start"], type="string", format="date",
            description="Historical recommendation feature cutoff, within the Olist data period"),
        "top_k": Param(SETTINGS["recommendation"]["top_k"], type="integer", minimum=1),
    },
    doc_md="""
    Manually prepare Olist data using worker Kaggle credentials; complete downloads
    are reused. Workers share the checkout and storage, so tasks run serially.
    Recommendation features describe one customer selected through Params.
    Commit the published snapshot metadata before triggering olist_model_training.
    """,
) as dag:
    steps = [
        snapshot_task("begin_preparation", "begin-preparation",
                      env={"OLIST_PARAMETERS": "{{ params | tojson }}"}),
        script_task("download_olist_data", "data_processing/download_olist_data.py",
                    "--output-dir", data_path("raw_dir"),
                    "--dataset", SETTINGS["data"]["dataset"]),
        script_task("validate_olist_data", "data_processing/validate_olist_data.py",
                    "--input-dir", data_path("raw_dir"),
                    "--schema", str(ROOT / "configs/schema.yaml")),
        snapshot_task("publish_raw_version", "raw"),
        script_task("clean_olist_data", "data_processing/clean_olist_data.py",
                    "--input-dir", data_path("raw_dir"),
                    "--output-dir", data_path("processed_dir")),
        script_task("prepare_curated_data", "data_processing/prepare_curated_data.py",
                    "--input-dir", data_path("processed_dir"),
                    "--output-dir", data_path("curated_dir"),
                    "--warehouse", data_path("warehouse")),
        script_task("build_forecasting_features", "features/build_forecasting_features.py",
                    "--input-dir", data_path("curated_dir"),
                    "--output-dir", data_path("forecasting_features"),
                    parameters={name.replace("_", "-"): name for name in SPLIT_DATES}),
        script_task("build_recommendation_features", "features/build_recommendation_features.py",
                    "--warehouse", data_path("warehouse"),
                    "--output-dir", data_path("recommendation_features"),
                    "--candidate-limit", str(SETTINGS["recommendation"]["candidate_limit"]),
                    parameters={"customer-id": "recommendation_customer_id",
                                "state": "recommendation_state", "as-of": "recommendation_as_of",
                                "top-k": "top_k"}),
        snapshot_task("publish_prepared_version", "prepared"),
    ]
    release = snapshot_task("release_workspace", "release-workspace", "--kind", "prepare",
                            trigger_rule="all_done")
    watcher = BashOperator(task_id="workflow_failed", bash_command="exit 1",
                           trigger_rule="one_failed", retries=0)
    chain(*steps)
    # Cleanup always runs; the final watcher keeps any upstream failure visible.
    for task in steps:
        task >> release
        task >> watcher
    release >> watcher
