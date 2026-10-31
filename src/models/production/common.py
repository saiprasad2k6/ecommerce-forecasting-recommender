"""Shared evaluation fingerprints, model publication and MLflow reporting."""

import hashlib
import json
from uuid import uuid4
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import joblib
import mlflow
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TRACKING_URI = f"sqlite:///{ROOT / 'data/baselines/mlflow.db'}"


def publish_model(model, path: Path, approved: bool) -> str:
    """Never replace an existing model with a rejected candidate.

    On a first run without an active artifact, a rejected candidate's selected
    baseline wrapper can be published as the initial fallback.
    """
    if not approved and path.exists():
        return "retained_existing_model"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(model, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return "published_trained_model" if approved else "published_baseline_fallback"


def evaluation_context(task, paths, **settings):
    """Identify the evaluation files, protocol and settings, independently of run time."""
    files = {}
    for path in paths:
        with Path(path).open("rb") as stream:
            files[str(Path(path).resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
    protocol = hashlib.sha256((Path(__file__).parents[1] / "evaluation.py").read_bytes()).hexdigest()
    context = {"task": task, "files": files, "protocol": protocol, **settings}
    context["evaluation_id"] = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    return context


def forecast_context(features_dir):
    return evaluation_context("forecasting", [features_dir / name for name in
        ["metadata.json", "validation.parquet", "test.parquet"]], target="target_demand_next_day")


def recommendation_context(warehouse, validation_cutoff, test_cutoff, test_end, candidate_limit, k):
    return evaluation_context("recommendation", [warehouse],
        validation_cutoff=str(pd.Timestamp(validation_cutoff).date()),
        test_cutoff=str(pd.Timestamp(test_cutoff).date()), test_end=str(pd.Timestamp(test_end).date()),
        candidate_limit=candidate_limit, k=k)


def save_run(model, report, params, output_dir, tracking_uri, experiment, comparison_id=None):
    """Log one independently evaluated model. Publication belongs to comparison."""
    (ROOT / "data/baselines").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=report["role"] + "__" + report["model"]) as run:
        output = Path(output_dir) / run.info.run_id
        output.mkdir(parents=True)
        report.update(comparison_id=comparison_id or uuid4().hex,
                      mlflow_run_id=run.info.run_id, output_dir=str(output), params=params)
        joblib.dump(model, output / "model.pkl")
        pd.DataFrame({split: report[split] for split in ["validation", "test"]}).T.to_csv(output / "metrics.csv")
        mlflow.log_params(params)
        mlflow.set_tags({key: report[key] for key in ["task", "role", "model", "comparison_id"]})
        mlflow.set_tag("evaluation_id", report["evaluation_context"]["evaluation_id"])
        for split, prefix in [("validation", "val"), ("test", "test")]:
            mlflow.log_metrics({prefix + "_" + key: value for key, value in report[split].items()
                                if key not in ["rows", "customers"]})
        mlflow.log_artifact(str(output / "model.pkl"))
        mlflow.log_artifact(str(output / "metrics.csv"))
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        mlflow.log_dict(report, "report.json")
    return report
