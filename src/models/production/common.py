"""Model approval, safe publication and MLflow reporting."""

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import joblib
import mlflow
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TRACKING_URI = f"sqlite:///{ROOT / 'data/baselines/mlflow.db'}"


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def atomic_dump(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(model, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_model(model, path: Path, approved: bool) -> str:
    """Never replace an existing model with a rejected candidate.

    On a first run without an active artifact, a rejected candidate's selected
    baseline wrapper can be published as the initial fallback.
    """
    if not approved and path.exists():
        return "retained_existing_model"
    atomic_dump(model, path)
    return "published_trained_model" if approved else "published_baseline_fallback"


def forecast_gate(candidate, baseline, varying=True):
    return bool(
        candidate["mae"] < baseline["mae"]
        and candidate["rmse"] <= baseline["rmse"]
        and varying
    )


def ranking_gate(candidate, baseline):
    return bool(
        candidate["recall_at_k"] > baseline["recall_at_k"]
        and candidate["precision_at_k"] >= baseline["precision_at_k"]
    )


def save_run(model, report, params, output_dir, model_path, tracking_uri, experiment):
    """Save the candidate and metrics, log to MLflow, then update the active model."""
    (ROOT / "data/baselines").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="production__" + report["candidate"]) as run:
        output = output_dir / run.info.run_id
        output.mkdir(parents=True)
        report.update(
            approved=report["validation_approved"] and report["holdout_passed"],
            publication="pending", output_dir=str(output), model_path=str(model_path),
            mlflow_run_id=run.info.run_id, params=params,
        )
        model.fallback = report["baseline"]
        joblib.dump(model, output / "candidate.pkl")
        model.use_model = report["approved"]
        metrics = pd.concat({
            split: pd.DataFrame.from_dict(report[split], orient="index")
            for split in ["validation", "test"]
        }, names=["split", "method"]).reset_index()
        metrics.to_csv(output / "metrics.csv", index=False)
        mlflow.log_params({"baseline_name": report["baseline"], **params})
        status = "validated_candidate" if report["approved"] else "rejected_candidate"
        mlflow.set_tags({
            "pipeline_integrated": "true",
            "selected_on_validation": str(report["validation_approved"]).lower(),
            "model_status": status,
        })
        for split, prefix in [("validation", "val"), ("test", "test")]:
            scores = pd.DataFrame.from_dict(report[split], orient="index").drop(
                columns=["rows", "customers"], errors="ignore"
            )
            values = {
                prefix: scores.loc[report["candidate"]].to_dict(),
                prefix + "_baseline": scores.loc[report["baseline"]].to_dict(),
            }
            flattened = pd.json_normalize(values, sep="_").iloc[0].dropna().to_dict()
            mlflow.log_metrics(flattened)
        mlflow.set_tags({
            "validation_gate_passed": str(report["validation_approved"]).lower(),
            "holdout_gate_passed": str(report["holdout_passed"]).lower(),
        })
        mlflow.log_artifact(str(output / "candidate.pkl"))
        mlflow.log_artifact(str(output / "metrics.csv"))

        # Publish only after logging succeeds, so failures keep the active model.
        report["publication"] = publish_model(model, model_path, report["approved"])
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        mlflow.log_dict(report, "report.json")
    return report
