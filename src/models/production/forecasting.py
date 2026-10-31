"""Train and log a next-day LightGBM candidate, independently of baselines."""
import argparse
import json
from pathlib import Path
import duckdb
from lightgbm import LGBMRegressor, early_stopping
import numpy as np
import pandas as pd
from src.models.baseline.forecasting_baselines import METHODS as BASELINES, predict_baseline
from src.models.evaluation import NUMERIC_FEATURES, READ_COLUMNS, feature_query as _feature_query, evaluate_forecasting
from src.models.production.common import ROOT, TRACKING_URI, forecast_context, save_run

class ProductionForecaster:
    """Persisted encoder, trained estimator and validation-selected fallback."""

    def __init__(self, estimator=None, encodings=None, fallback="last_day", use_model=True):
        self.estimator = estimator
        self.encodings = encodings or {}
        self.fallback = fallback
        self.use_model = use_model

    def prepare_features(self, frame):
        result = frame[NUMERIC_FEATURES].astype("float32")
        target_date = pd.to_datetime(frame.purchase_date) + pd.Timedelta(days=1)
        result["target_day_of_week"] = target_date.dt.dayofweek.astype("float32")
        result["target_month"] = target_date.dt.month.astype("float32")
        result["target_day_of_month"] = target_date.dt.day.astype("float32")
        for column in ["product_id", "customer_state"]:
            codes = pd.Index(self.encodings[column]).get_indexer(frame[column])
            result[column + "_code"] = codes.astype("float32")
        return result

    def predict(self, frame):
        if not self.use_model:
            return predict_baseline(frame, self.fallback)
        return np.maximum(self.estimator.predict(self.prepare_features(frame)), 0)

    encode = prepare_features  # Keep existing notebook examples working.


def prepare_data(features_dir, max_rows=1_000_000, seed=42):
    """Read split dates and prepare weighted training and validation samples."""
    if max_rows < 2:
        raise ValueError("max-train-rows must be at least 2")
    metadata = json.loads((features_dir / "metadata.json").read_text())
    ranges = metadata["date_ranges"]
    validation_start = pd.Timestamp(ranges["validation"]["min"])
    test_start = pd.Timestamp(ranges["test"]["min"])
    if (
        pd.Timestamp(ranges["train"]["max"]) >= validation_start
        or pd.Timestamp(ranges["validation"]["max"]) >= test_start
    ):
        raise ValueError("Feature splits overlap or are not chronological")

    base = _feature_query(features_dir, "train", validation_start)
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        total, positive = db.sql(f"""
            SELECT COUNT(*), COUNT(*) FILTER (WHERE target_demand_next_day > 0)
            FROM ({base})
        """).fetchone()
        if not positive:
            raise ValueError("Training history has no positive next-day targets")
        positive_count = min(positive, max_rows // 2)
        zero_count = min(total - positive, max_rows - positive_count)
        # Each entry describes a target filter, its sample size and its population.
        # strata = [
        #     (">", 200, 200),      # Sample 200 positive rows from 200 available
        #     ("=", 800, 9_800),    # Sample 800 zero rows from 9,800 available
        # ]
        strata = [(">", positive_count, positive), ("=", zero_count, total - positive)]
        frames, weights = [], []
        for operator, count, population in strata:
            if count == 0:
                continue
            sampled = db.sql(f"""
                SELECT * FROM (
                    SELECT * FROM ({base}) WHERE target_demand_next_day {operator} 0
                ) USING SAMPLE reservoir({count} ROWS) REPEATABLE ({seed})
            """).df()
            frames.append(sampled)
            weights.extend([population / len(sampled)] * len(sampled))
        query = _feature_query(features_dir, "validation", test_start)
        validation = db.sql(f"""
            SELECT * FROM ({query})
            USING SAMPLE reservoir(200000 ROWS) REPEATABLE ({seed})
        """).df()
    if validation.empty:
        raise ValueError("Validation is empty after purging its label boundary")
    train = pd.concat(frames, ignore_index=True)
    return train, validation, np.asarray(weights), metadata, total


def train_model(train, validation, weights, n_estimators=100, seed=42):
    """Fit a next-day forecaster; validation MAE controls early stopping."""
    if n_estimators < 1:
        raise ValueError("n-estimators must be positive")
    encodings = {
        column: pd.Index(sorted(train[column].unique()))
        for column in ["product_id", "customer_state"]
    }
    estimator = LGBMRegressor(
        objective="poisson", metric="mae", n_estimators=n_estimators,
        random_state=seed, n_jobs=2, verbosity=-1,
    )
    model = ProductionForecaster(estimator, encodings)
    estimator.fit(
        model.prepare_features(train), train.target_demand_next_day,
        sample_weight=weights,
        categorical_feature=["product_id_code", "customer_state_code"],
        eval_X=model.prepare_features(validation),
        eval_y=validation.target_demand_next_day,
        callbacks=[early_stopping(20, verbose=False)],
    )
    return model


def evaluate_model(model, features_dir, split, label_before=None):
    metrics, varying = evaluate_forecasting({"lightgbm": model}, features_dir, split, label_before)
    return metrics, varying["lightgbm"]


def run_pipeline(features_dir=ROOT / "data/features/forecasting", output_dir=ROOT / "data/production/forecasting",
                 tracking_uri=TRACKING_URI, max_train_rows=1_000_000, n_estimators=100, seed=42, comparison_id=None):
    context = forecast_context(features_dir)
    train, validation, weights, metadata, population = prepare_data(features_dir, max_train_rows, seed)
    model = train_model(train, validation, weights, n_estimators, seed)
    boundary = pd.Timestamp(metadata["date_ranges"]["test"]["min"])
    validation_metrics, varying = evaluate_model(model, features_dir, "validation", boundary)
    test_metrics, _ = evaluate_model(model, features_dir, "test")
    if context != forecast_context(features_dir):
        raise ValueError("Evaluation data changed during training")
    report = {"task": "forecasting", "role": "candidate", "model": "lightgbm",
              "validation": validation_metrics["lightgbm"], "test": test_metrics["lightgbm"],
              "prediction_varies": varying, "evaluation_context": context}
    params = {"target": "target_demand_next_day", "training_rows": len(train),
              "training_population_rows": population, "max_train_rows": max_train_rows,
              "seed": seed, "n_estimators": n_estimators, "trained_estimators": model.estimator.n_estimators_}
    return save_run(model, report, params, output_dir, tracking_uri, "ecommerce-demand-forecasting", comparison_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=ROOT / "data/features/forecasting")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/production/forecasting")
    parser.add_argument("--comparison-id")
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_pipeline(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
