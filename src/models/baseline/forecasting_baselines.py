"""Evaluate next-day forecasting rules and log independent MLflow runs."""
import argparse
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.models.evaluation import evaluate_forecasting
from src.models.production.common import ROOT, TRACKING_URI, forecast_context, save_run

METHODS = ["last_day", "same_day_last_week", "moving_average_7", "moving_average_28", "seasonal_naive_28", "always_zero"]

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
    METHODS = dict.fromkeys(METHODS)

    def __init__(self, method="moving_average_28"):
        if method not in METHODS:
            raise ValueError(f"Unknown baseline: {method}")
        self.method = method

    def predict(self, frame):
        return predict_baseline(frame, self.method)

    def evaluate(self, frame, target_col="target_demand_next_day"):
        actual, predicted = frame[target_col], self.predict(frame)
        return {"method": self.method, "mae": mean_absolute_error(actual, predicted),
                "rmse": root_mean_squared_error(actual, predicted)}


def evaluate_baselines_duckdb(parquet_path, split_name):
    metadata = json.loads((parquet_path.parent / "metadata.json").read_text())
    boundary = pd.Timestamp(metadata["date_ranges"]["test"]["min"]) if split_name == "validation" else None
    metrics, _ = evaluate_forecasting({name: BaselineForecaster(name) for name in METHODS},
                                     parquet_path.parent, split_name, boundary)
    return pd.DataFrame.from_dict(metrics, orient="index").rename_axis("baseline").reset_index().assign(split=split_name).sort_values("mae")


def run_baseline_pipeline(features_dir=ROOT / "data/features/forecasting",
                          output_dir=ROOT / "data/baselines/forecasting", tracking_uri=TRACKING_URI,
                          experiment_name="ecommerce-demand-forecasting", comparison_id=None):
    comparison_id = comparison_id or uuid4().hex
    context = forecast_context(features_dir)
    metadata = json.loads((features_dir / "metadata.json").read_text())
    boundary = pd.Timestamp(metadata["date_ranges"]["test"]["min"])
    models = {name: BaselineForecaster(name) for name in METHODS}
    validation, _ = evaluate_forecasting(models, features_dir, "validation", boundary)
    test, _ = evaluate_forecasting(models, features_dir, "test")
    if context != forecast_context(features_dir):
        raise ValueError("Evaluation data changed during baseline evaluation")
    runs = [save_run(model, {"task": "forecasting", "role": "baseline", "model": name,
                 "validation": validation[name], "test": test[name], "evaluation_context": context},
                 {"target": "target_demand_next_day"}, output_dir, tracking_uri, experiment_name, comparison_id)
            for name, model in models.items()]
    champion = min((name for name in METHODS if name != "always_zero"), key=lambda name: (validation[name]["mae"], name))
    return {"comparison_id": comparison_id, "champion_baseline": champion, "runs": runs}


run_pipeline = run_baseline_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=ROOT / "data/features/forecasting")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/baselines/forecasting")
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default="ecommerce-demand-forecasting")
    parser.add_argument("--comparison-id")
    print(json.dumps(run_pipeline(**vars(parser.parse_args())), indent=2))
    return 0
