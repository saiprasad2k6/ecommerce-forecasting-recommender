"""Compare matching MLflow runs and publish approved models or baseline fallbacks."""
import argparse
import json
import math
from pathlib import Path
import tempfile

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.models.baseline.forecasting_baselines import METHODS as FORECAST_BASELINES
from src.models.baseline.recommendation_baselines import METHODS as REC_BASELINES
from src.models.production.common import ROOT, TRACKING_URI, publish_model

TASKS = {
    "forecasting": ("ecommerce-demand-forecasting", "lightgbm", FORECAST_BASELINES, "forecast_model.pkl"),
    "recommendation": ("ecommerce-recommendations", "lightgbm_ranker", REC_BASELINES, "recommender.pkl"),
}


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


def compare_models(comparison_id, tracking_uri=TRACKING_URI, models_dir=ROOT / "models", output_dir=ROOT / "data/comparisons"):
    """Use only completed runs from this group, with identical evaluation data."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    decisions, selected_models = {}, {}
    # Validate both tasks and load both artifacts before modifying any active model.
    with tempfile.TemporaryDirectory() as temporary:
        for task, (experiment_name, candidate_name, baseline_names, filename) in TASKS.items():
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Missing experiment: {experiment_name}")
            # Iterate all pages; retries can create multiple completed runs. The newest
            # successful run for each model in this exact comparison group wins.
            runs, token = [], None
            while True:
                page = client.search_runs([experiment.experiment_id],
                    filter_string="attributes.status = 'FINISHED'", order_by=["attributes.start_time DESC"],
                    max_results=1000, page_token=token)
                runs.extend(page)
                token = page.token
                if not token:
                    break
            reports = {}
            for run in runs:
                tags = run.data.tags
                name = tags.get("model")
                expected_role = "candidate" if name == candidate_name else "baseline"
                if (tags.get("comparison_id") != comparison_id or tags.get("task") != task
                    or tags.get("role") != expected_role or name not in [candidate_name, *baseline_names]
                    or name in reports):
                    continue
                path = client.download_artifacts(run.info.run_id, "report.json", temporary)
                report = json.loads(Path(path).read_text())
                if (report["mlflow_run_id"] != run.info.run_id or report["comparison_id"] != comparison_id
                    or report["model"] != name or report["role"] != expected_role or report["task"] != task):
                    raise ValueError("Run metadata does not match its report")
                reports[name] = report
            if set(reports) != {candidate_name, *baseline_names}:
                raise ValueError(f"Incomplete {task} comparison group {comparison_id}: missing "
                                 f"{sorted({candidate_name, *baseline_names} - set(reports))}")
            candidate = reports[candidate_name]
            for report in reports.values():
                if report["evaluation_context"] != candidate["evaluation_context"]:
                    raise ValueError(f"Mismatched evaluation data/settings for {task}")
                for split in ["validation", "test"]:
                    scores = report[split]
                    count = "rows" if task == "forecasting" else "customers"
                    if scores[count] <= 0 or scores[count] != candidate[split][count]:
                        raise ValueError(f"Mismatched evaluation rows for {task}")
                    keys = ["mae", "rmse"] if task == "forecasting" else ["precision_at_k", "recall_at_k"]
                    if any(not math.isfinite(scores[key]) or scores[key] < 0 for key in keys):
                        raise ValueError(f"Invalid evaluation metrics for {task}")
            eligible = [name for name in baseline_names if name != "always_zero"]
            if task == "forecasting":
                baseline_name = min(eligible, key=lambda name: (reports[name]["validation"]["mae"], name))
                gate = forecast_gate
            else:
                baseline_name = min(eligible, key=lambda name: (-reports[name]["validation"]["recall_at_k"],
                    -reports[name]["validation"]["precision_at_k"], name))
                gate = ranking_gate
            baseline = reports[baseline_name]
            validation_passed = gate(candidate["validation"], baseline["validation"])
            if task == "forecasting":
                validation_passed = validation_passed and candidate["prediction_varies"]
            holdout_passed = gate(candidate["test"], baseline["test"])
            approved = bool(validation_passed and holdout_passed)
            chosen = candidate if approved else baseline
            artifact = client.download_artifacts(chosen["mlflow_run_id"], "model.pkl", temporary)
            selected_models[task] = joblib.load(artifact)
            decisions[task] = {
                "comparison_id": comparison_id, "task": task, "candidate": candidate_name,
                "baseline": baseline_name, "candidate_run_id": candidate["mlflow_run_id"],
                "baseline_run_id": baseline["mlflow_run_id"], "validation_approved": bool(validation_passed),
                "holdout_passed": bool(holdout_passed), "approved": approved,
                "validation": {"candidate": candidate["validation"], "baseline": baseline["validation"]},
                "test": {"candidate": candidate["test"], "baseline": baseline["test"]},
                "evaluation_id": candidate["evaluation_context"]["evaluation_id"],
                "model_path": str(Path(models_dir) / filename),
            }
    for task, report in decisions.items():
        mlflow.set_experiment(TASKS[task][0])
        with mlflow.start_run(run_name="comparison") as run:
            mlflow.set_tags({"role": "comparison", "task": task, "comparison_id": comparison_id,
                            "candidate_run_id": report["candidate_run_id"],
                            "baseline_run_id": report["baseline_run_id"], "approved": str(report["approved"]).lower(),
                            "evaluation_id": report["evaluation_id"],
                            "validation_gate_passed": str(report["validation_approved"]).lower(),
                            "holdout_gate_passed": str(report["holdout_passed"]).lower()})
            # Record the decision before publication; failed logging preserves active files.
            mlflow.log_dict(report, "decision.json")
            report["publication"] = publish_model(selected_models[task], Path(report["model_path"]), report["approved"])
            report["comparison_run_id"] = run.info.run_id
            output = Path(output_dir) / run.info.run_id
            output.mkdir(parents=True)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            mlflow.log_dict(report, "report.json")
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/comparisons")
    print(json.dumps(compare_models(**vars(parser.parse_args())), indent=2))
    return 0
