"""Compare matching MLflow runs and publish approved models or baseline fallbacks."""
import argparse
import json
from pathlib import Path
import tempfile

import joblib
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from src.models.baseline.forecasting_baselines import METHODS as FORECAST_BASELINES
from src.models.baseline.recommendation_baselines import METHODS as REC_BASELINES
from src.models.production.common import ROOT, TRACKING_URI, publish_model

TASKS = {
    "forecasting": {
        "experiment": "ecommerce-demand-forecasting",
        "candidate": "lightgbm",
        "baselines": FORECAST_BASELINES,
        "filename": "forecast_model.pkl",
    },
    "recommendation": {
        "experiment": "ecommerce-recommendations",
        "candidate": "lightgbm_ranker",
        "baselines": REC_BASELINES,
        "filename": "recommender.pkl",
    },
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


def load_model_results(task, pipeline_run_id):
    """Load the newest completed run for each required model in this group."""
    config = TASKS[task]
    runs = mlflow.search_runs(
        experiment_names=[config["experiment"]],
        filter_string=(f'attributes.status = "FINISHED" '
                       f'AND tags.pipeline_run_id = {json.dumps(pipeline_run_id)} '
                       f'AND tags.task = {json.dumps(task)}'),
        order_by=["attributes.start_time DESC"], output_format="list",
    )
    required = {config["candidate"], *config["baselines"]}
    model_results = {}
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as temporary:
        for run in runs:
            name = run.data.tags.get("model")
            role = "candidate" if name == config["candidate"] else "baseline"
            if name not in required or name in model_results or run.data.tags.get("role") != role:
                continue
            path = client.download_artifacts(run.info.run_id, "report.json", temporary)
            model_results[name] = json.loads(Path(path).read_text())
            # Use the ID from MLflow to retrieve this run's model artifact later.
            model_results[name]["mlflow_run_id"] = run.info.run_id

    missing = required - model_results.keys()
    if missing:
        raise ValueError(f"Incomplete {task} comparison group {pipeline_run_id}: missing {sorted(missing)}")
    return model_results


def compare_model_results(task, model_results):
    """Choose the validation baseline and check the candidate on both splits."""
    config = TASKS[task]
    candidate = model_results[config["candidate"]]
    count = "rows" if task == "forecasting" else "customers"
    for model_result in model_results.values():
        if model_result["evaluation_context"] != candidate["evaluation_context"]:
            raise ValueError(f"Mismatched evaluation data/settings for {task}")
        for split in ["validation", "test"]:
            if model_result[split][count] != candidate[split][count]:
                raise ValueError(f"Mismatched evaluation rows for {task}")

    scores = pd.DataFrame.from_dict(
        {name: model_results[name]["validation"] for name in config["baselines"] if name != "always_zero"},
        orient="index",
    ).rename_axis("model").reset_index()
    if task == "forecasting":
        scores = scores.sort_values(["mae", "model"])
        gate = forecast_gate
    else:
        scores = scores.sort_values(["recall_at_k", "precision_at_k", "model"], ascending=[False, False, True])
        gate = ranking_gate
    baseline_name = scores.iloc[0]["model"]
    baseline = model_results[baseline_name]

    validation_passed = gate(candidate["validation"], baseline["validation"])
    if task == "forecasting":
        validation_passed = validation_passed and candidate["prediction_varies"]
    holdout_passed = gate(candidate["test"], baseline["test"])
    return {
        "task": task, "candidate": config["candidate"], "baseline": baseline_name,
        "candidate_run_id": candidate["mlflow_run_id"], "baseline_run_id": baseline["mlflow_run_id"],
        "validation_approved": bool(validation_passed), "holdout_passed": bool(holdout_passed),
        "approved": bool(validation_passed and holdout_passed),
        "validation": {"candidate": candidate["validation"], "baseline": baseline["validation"]},
        "test": {"candidate": candidate["test"], "baseline": baseline["test"]},
        "evaluation_id": candidate["evaluation_context"]["evaluation_id"],
    }


def save_comparison(decision, model, output_dir):
    """Log the decision, publish the selected model, and save the final report."""
    mlflow.set_experiment(TASKS[decision["task"]]["experiment"])
    with mlflow.start_run(run_name="comparison") as run:
        mlflow.set_tags({
            "role": "comparison", "task": decision["task"], "pipeline_run_id": decision["pipeline_run_id"],
            "candidate_run_id": decision["candidate_run_id"], "baseline_run_id": decision["baseline_run_id"],
            "evaluation_id": decision["evaluation_id"], "approved": str(decision["approved"]).lower(),
            "validation_gate_passed": str(decision["validation_approved"]).lower(),
            "holdout_gate_passed": str(decision["holdout_passed"]).lower(),
        })
        # Record the decision before publication; failed logging preserves active files.
        mlflow.log_dict(decision, "decision.json")
        decision["publication"] = publish_model(model, Path(decision["model_path"]), decision["approved"])
        decision["comparison_run_id"] = run.info.run_id
        output = Path(output_dir) / run.info.run_id
        output.mkdir(parents=True)
        (output / "report.json").write_text(json.dumps(decision, indent=2, allow_nan=False))
        mlflow.log_dict(decision, "report.json")


def compare_models(pipeline_run_id, tracking_uri=TRACKING_URI, models_dir=ROOT / "models", output_dir=ROOT / "data/comparisons"):
    """Load model results, compare models, then save both decisions."""
    mlflow.set_tracking_uri(tracking_uri)
    decisions, models = {}, {}
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as temporary:
        for task, config in TASKS.items():
            model_results = load_model_results(task, pipeline_run_id)
            decision = compare_model_results(task, model_results)
            decision["pipeline_run_id"] = pipeline_run_id
            decision["model_path"] = str(Path(models_dir) / config["filename"])
            run_id = decision["candidate_run_id"] if decision["approved"] else decision["baseline_run_id"]
            artifact = client.download_artifacts(run_id, "model.pkl", temporary)
            models[task] = joblib.load(artifact)
            decisions[task] = decision

    # Both comparisons and artifact loads must succeed before publishing either model.
    for task, decision in decisions.items():
        save_comparison(decision, models[task], output_dir)
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-run-id", required=True)
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/comparisons")
    print(json.dumps(compare_models(**vars(parser.parse_args())), indent=2))
    return 0
