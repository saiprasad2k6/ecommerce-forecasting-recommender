"""Compare matching MLflow runs and publish approved models or baseline fallbacks."""
import argparse
import json
from pathlib import Path
import tempfile

import mlflow
from mlflow.tracking import MlflowClient
import joblib
import pandas as pd

from src.models.forecasting.forecasting_baseline import METHODS as FORECAST_BASELINES
from src.models.recommendation.recommendation_baseline import METHODS as REC_BASELINES
from src.models.common import ROOT, TRACKING_URI, MODEL_FILES, publish_model, set_experiment

MODEL_JOBS = json.loads((ROOT / "configs/production_models.json").read_text())

TASKS = {
    "forecasting": {
        "experiment": "ecommerce-demand-forecasting",
        "candidates": list(MODEL_JOBS["forecasting"]),
        "baselines": FORECAST_BASELINES,
    },
    "recommendation": {
        "experiment": "ecommerce-recommendations",
        "candidates": list(MODEL_JOBS["recommendation"]),
        "baselines": REC_BASELINES,
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
    required = {*config["candidates"], *config["baselines"]}
    model_results = {}
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as temporary:
        for run in runs:
            name = run.data.tags.get("model")
            role = "candidate" if name in config["candidates"] else "baseline"
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
    """Choose a model on validation; test can only veto that winner's release."""
    config = TASKS[task]
    reference = model_results[config["candidates"][0]]
    count = "rows" if task == "forecasting" else "customers"
    for result in model_results.values():
        if result["evaluation_context"] != reference["evaluation_context"]:
            raise ValueError(f"Mismatched evaluation data/settings for {task}")
        for split in ["validation", "test"]:
            if result[split][count] != reference[split][count]:
                raise ValueError(f"Mismatched evaluation rows for {task}")

    scores = pd.DataFrame.from_dict(
        {name: result["validation"] for name, result in model_results.items() if name != "always_zero"},
        orient="index",
    ).rename_axis("model").reset_index()
    if task == "forecasting":
        scores = scores.sort_values(["mae", "rmse", "model"])
        gate = forecast_gate
    else:
        scores = scores.sort_values(["recall_at_k", "precision_at_k", "model"], ascending=[False, False, True])
        gate = ranking_gate
    baseline_name = scores.loc[scores.model.isin(config["baselines"])].iloc[0]["model"]
    baseline = model_results[baseline_name]
    candidate_names = scores.loc[scores.model.isin(config["candidates"])].model.tolist()
    eligible = [
        name for name in candidate_names
        if gate(model_results[name]["validation"], baseline["validation"])
        and (task != "forecasting" or model_results[name]["prediction_varies"])
    ]
    candidate_name = eligible[0] if eligible else candidate_names[0]
    candidate = model_results[candidate_name]
    holdout_passed = gate(candidate["test"], baseline["test"])
    return {
        "task": task, "candidate": candidate_name, "baseline": baseline_name,
        "candidate_run_id": candidate["mlflow_run_id"], "baseline_run_id": baseline["mlflow_run_id"],
        "candidate_run_ids": {name: model_results[name]["mlflow_run_id"] for name in candidate_names},
        "validation_ranking": candidate_names, "validation_eligible": eligible,
        "validation_approved": bool(eligible), "holdout_passed": bool(holdout_passed),
        "approved": bool(eligible and holdout_passed),
        "validation": {"candidate": candidate["validation"], "baseline": baseline["validation"]},
        "test": {"candidate": candidate["test"], "baseline": baseline["test"]},
        "evaluation_id": candidate["evaluation_context"]["evaluation_id"],
        "inference_parameters": {key: reference["evaluation_context"][key]
                                 for key in ["k", "candidate_limit"] if key in reference["evaluation_context"]},
    }


def save_comparison(decision, model, output_dir):
    """Log the decision, publish the selected model, and save the final report."""
    set_experiment(TASKS[decision["task"]]["experiment"])
    with mlflow.start_run(run_name="comparison") as run:
        mlflow.set_tags({
            "role": "comparison", "task": decision["task"], "pipeline_run_id": decision["pipeline_run_id"],
            "candidate": decision["candidate"], "baseline": decision["baseline"],
            "candidate_run_id": decision["candidate_run_id"], "baseline_run_id": decision["baseline_run_id"],
            "evaluation_id": decision["evaluation_id"], "approved": str(decision["approved"]).lower(),
            "validation_gate_passed": str(decision["validation_approved"]).lower(),
            "holdout_gate_passed": str(decision["holdout_passed"]).lower(),
        })
        # Record the decision before publication; failed logging preserves active files.
        mlflow.log_dict(decision, "decision.json")
        selection = {
            "task": decision["task"], "approved_for_inference": True,
            "model_name": decision["candidate"] if decision["approved"] else decision["baseline"],
            "model_role": "candidate" if decision["approved"] else "baseline",
            "model_run_id": decision["candidate_run_id"] if decision["approved"] else decision["baseline_run_id"],
            "training_pipeline_run_id": decision["pipeline_run_id"],
            "comparison_run_id": run.info.run_id,
            "evaluation_id": decision["evaluation_id"],
            **decision["inference_parameters"],
        }
        decision["publication"] = publish_model(
            model, Path(decision["model_path"]), decision["approved"], metadata=selection,
        )
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
        for task in TASKS:
            model_results = load_model_results(task, pipeline_run_id)
            decision = compare_model_results(task, model_results)
            decision["pipeline_run_id"] = pipeline_run_id
            decision["model_path"] = str(Path(models_dir) / MODEL_FILES[task])
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
