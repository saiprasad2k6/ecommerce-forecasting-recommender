"""Evaluate next-day forecasting rules and log independent MLflow runs."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.models.evaluation import evaluate_forecasting as evaluate_model
from src.models.common import ROOT, TRACKING_URI, forecast_context, save_run

METHODS = [
    "last_day", "same_day_last_week", "moving_average_7",
    "moving_average_28", "seasonal_naive_28", "always_zero",
]


def predict_baseline(frame, method):
    """All rules forecast t+1 from observations through t, not y(t) from t-1."""
    if method == "last_day":
        return frame.demand.to_numpy(float)
    if method == "same_day_last_week":
        return frame.demand_lag_6.to_numpy(float)
    if method == "seasonal_naive_28":
        return frame.demand_lag_27.to_numpy(float)
    if method == "always_zero":
        return np.zeros(len(frame))
    days = {"moving_average_7": 7, "moving_average_28": 28}[method]
    # Shift the original past-only window forward by one day using known values.
    past_total = frame[f"rolling_mean_{days}"] * days
    updated_total = past_total + frame.demand - frame[f"demand_lag_{days}"]
    return (updated_total / days).to_numpy(float)


class BaselineForecaster:
    def __init__(self, method="moving_average_28"):
        if method not in METHODS:
            raise ValueError(f"Unknown baseline: {method}")
        self.method = method

    def predict(self, frame):
        return predict_baseline(frame, self.method)


def run_pipeline(
    features_dir=ROOT / "data/features/forecasting",
    output_dir=ROOT / "data/baselines/forecasting",
    tracking_uri=TRACKING_URI,
    pipeline_run_id=None,
):
    pipeline_run_id = pipeline_run_id or uuid4().hex
    context = forecast_context(features_dir)
    metadata = json.loads((features_dir / "metadata.json").read_text())
    boundary = pd.Timestamp(metadata["boundaries"]["test_start"])
    models = {name: BaselineForecaster(name) for name in METHODS}

    validation, _ = evaluate_model(models, features_dir, "validation", boundary)
    test, _ = evaluate_model(models, features_dir, "test")
    if context != forecast_context(features_dir):
        raise ValueError("Evaluation data changed during baseline evaluation")

    runs = []
    for name, model in models.items():
        report = {
            "task": "forecasting", "role": "baseline", "model": name,
            "validation": validation[name], "test": test[name],
            "evaluation_context": context,
        }
        runs.append(save_run(
            model, report, {"target": "target_demand_next_day"},
            output_dir, tracking_uri, "ecommerce-demand-forecasting", pipeline_run_id,
        ))
    return {"pipeline_run_id": pipeline_run_id, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=ROOT / "data/features/forecasting"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/baselines/forecasting"
    )
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--pipeline-run-id")
    report = run_pipeline(**vars(parser.parse_args()))
    print(f"Pipeline run: {report['pipeline_run_id']}")
    return 0
