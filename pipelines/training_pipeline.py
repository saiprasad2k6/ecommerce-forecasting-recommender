"""Manual production-candidate training DAG; run data_pipeline.py first."""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shlex
import sys

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param


ROOT = Path(os.environ.get("OLIST_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
PROJECT_PYTHON = ROOT / ".venv/bin/python"
DEFAULT_PYTHON = str(PROJECT_PYTHON) if PROJECT_PYTHON.is_file() else sys.executable
PYTHON = os.environ.get("OLIST_PYTHON", DEFAULT_PYTHON)


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
    dag_id="olist_production_training",
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
    tags=["olist", "production", "training"],
    params={
        "max_train_rows": Param(1_000_000, type="integer", minimum=2),
        "forecast_estimators": Param(100, type="integer", minimum=1),
        "ranking_estimators": Param(100, type="integer", minimum=1),
        "candidate_limit": Param(100, type="integer", minimum=10),
        "top_k": Param(10, type="integer", minimum=1),
    },
    doc_md="""
    Run `olist_end_to_end_pipeline` successfully before triggering this DAG.
    Baseline jobs and candidate jobs log separate MLflow runs sharing this DAG run ID.
    The final comparison task verifies evaluation fingerprints and publishes approved
    candidates, or an initial baseline fallback. Tasks run serially for SQLite/DuckDB.
    This DAG writes local artifacts; it does not
    deploy an API or download/rebuild input data. Validation chooses whether to
    select a trained candidate; a frozen holdout check can veto its publication.
    Failed candidates keep the existing model/baseline fallback.
    Check each versioned report's `approved` and `publication` fields: a successful
    DAG run does not imply that both trained candidates passed their gates.
    Recommendation defaults use March–June 2018 training, July validation and August test.
    """,
) as dag:
    validate_inputs = training_task(
        "validate_training_inputs",
        "production/validate_inputs.py",
        '--candidate-limit "$CANDIDATES" --top-k "$TOP_K"',
        env={
            "CANDIDATES": "{{ params.candidate_limit }}",
            "TOP_K": "{{ params.top_k }}",
        },
    )
    forecast_baselines = training_task(
        "evaluate_forecasting_baselines", "baseline/forecasting_baselines.py",
        '--pipeline-run-id "$PIPELINE_RUN_ID"', env={},
    )
    recommendation_baselines = training_task(
        "evaluate_recommendation_baselines", "baseline/recommendation_baselines.py",
        '--pipeline-run-id "$PIPELINE_RUN_ID" --candidate-limit "$CANDIDATES" --top-k "$TOP_K"',
        env={"CANDIDATES": "{{ params.candidate_limit }}", "TOP_K": "{{ params.top_k }}"},
    )
    forecasting = training_task(
        "train_forecasting_model",
        "production/train_forecasting.py",
        '--pipeline-run-id "$PIPELINE_RUN_ID" --max-train-rows "$TRAIN_ROWS" --n-estimators "$ESTIMATORS"',
        env={
            "TRAIN_ROWS": "{{ params.max_train_rows }}",
            "ESTIMATORS": "{{ params.forecast_estimators }}",
        },
    )
    recommendation = training_task(
        "train_recommendation_model",
        "production/train_recommendation.py",
        '--pipeline-run-id "$PIPELINE_RUN_ID" --candidate-limit "$CANDIDATES" --top-k "$TOP_K" --n-estimators "$ESTIMATORS"',
        env={
            "CANDIDATES": "{{ params.candidate_limit }}",
            "TOP_K": "{{ params.top_k }}",
            "ESTIMATORS": "{{ params.ranking_estimators }}",
        },
    )
    comparison = training_task(
        "compare_models", "comparison/compare_models.py",
        '--pipeline-run-id "$PIPELINE_RUN_ID"', env={},
    )
    validate_inputs >> forecast_baselines >> recommendation_baselines >> forecasting >> recommendation >> comparison
