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
    description="Validate inputs, evaluate baselines, train candidates, then compare matching MLflow runs",
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
        **{name: Param(REPLAY[name], type="string", format="date") for name in
           ["training_start", "validation_start", "test_start", "monitoring_start"]},
        "max_train_rows": Param(1_000_000, type="integer", minimum=2),
        "forecast_estimators": Param(100, type="integer", minimum=1),
        "ranking_estimators": Param(100, type="integer", minimum=1),
        "candidate_limit": Param(SETTINGS["recommendation"]["candidate_limit"], type="integer", minimum=10),
        "top_k": Param(SETTINGS["recommendation"]["top_k"], type="integer", minimum=1),
    },
    doc_md="""
    Run `olist_data_preparation` successfully before triggering this DAG.
    Baseline jobs and candidate jobs log separate MLflow runs sharing this DAG run ID.
    Model jobs are registered in configs/production_models.json. Validation ranks
    all configured candidates; test only checks the chosen validation winner.
    The final comparison task verifies evaluation fingerprints and publishes approved
    candidates, or an initial baseline fallback. Tasks run serially for SQLite/DuckDB.
    This DAG writes local artifacts; it does not
    deploy an API or download/rebuild input data. Validation chooses whether to
    select a trained candidate; a frozen holdout check can veto its publication.
    Failed candidates keep the existing model/baseline fallback.
    Check each versioned report's `approved` and `publication` fields: a successful
    DAG run does not imply that both trained candidates passed their gates.
    Recommendation defaults use March–May 2018 training, June validation and July test.
    August is reserved for inference and monitoring.
    """,
) as dag:
    validate_inputs = training_task(
        "validate_training_inputs",
        "validate_inputs.py",
        '--candidate-limit "$CANDIDATES" --top-k "$TOP_K"',
        env={
            "CANDIDATES": "{{ params.candidate_limit }}",
            "TOP_K": "{{ params.top_k }}",
        },
    )
    forecast_baselines = training_task(
        "evaluate_forecasting_baselines", "forecasting/forecasting_baseline.py",
        '--pipeline-run-id "$PIPELINE_RUN_ID"', env={},
    )
    recommendation_baselines = training_task(
        "evaluate_recommendation_baselines", "recommendation/recommendation_baseline.py",
        '--pipeline-run-id "$PIPELINE_RUN_ID" --candidate-limit "$CANDIDATES" --top-k "$TOP_K"'
        ' --validation-cutoff "$VALIDATION_START" --test-cutoff "$TEST_START" --test-end "$MONITORING_START"',
        env={"CANDIDATES": "{{ params.candidate_limit }}", "TOP_K": "{{ params.top_k }}",
             **{name.upper(): "{{ params." + name + " }}" for name in
                ["validation_start", "test_start", "monitoring_start"]}},
    )
    previous = recommendation_baselines
    for task, models in MODEL_JOBS.items():
        for model_name, job in models.items():
            arguments = '--pipeline-run-id "$PIPELINE_RUN_ID"'
            env = {}
            for flag, parameter in job["parameters"].items():
                env[parameter.upper()] = "{{ params." + parameter + " }}"
                arguments += f' --{flag} "${parameter.upper()}"'
            training = training_task(
                f"train_{task}_{model_name}", job["script"], arguments, env,
            )
            previous >> training
            previous = training
    comparison = training_task(
        "compare_models", "comparison/compare_models.py",
        '--pipeline-run-id "$PIPELINE_RUN_ID"', env={},
    )
    validate_inputs >> forecast_baselines >> recommendation_baselines
    previous >> comparison
