"""Train a next-day LightGBM forecaster with chronological baseline validation."""

import argparse
import json
from pathlib import Path

import duckdb
from lightgbm import LGBMRegressor, early_stopping
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.production.common import (
    ROOT, TRACKING_URI, forecast_gate, save_run, sql_path,
)


NUMERIC_FEATURES = [
    "demand", "demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_28",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_28",
]
READ_COLUMNS = [
    "purchase_date", "product_id", "customer_state", *NUMERIC_FEATURES,
    "demand_lag_6", "demand_lag_27", "target_demand_next_day",
]
BASELINES = [
    "last_day", "same_day_last_week", "moving_average_7", "moving_average_28",
    "seasonal_naive_28", "always_zero",
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


def _feature_query(features_dir, split, label_before=None):
    """Read prepared features, excluding labels from the next period when needed."""
    path = sql_path(features_dir / f"{split}.parquet")
    query = f"""
        SELECT {', '.join(READ_COLUMNS)}
        FROM read_parquet('{path}')
    """
    if label_before is not None:
        query += f"WHERE purchase_date + INTERVAL 1 DAY < DATE '{label_before.date()}'"
    return query


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
    """Evaluate every prepared row in batches, without loading the entire split."""
    batches = []
    prediction_min, prediction_max = float("inf"), float("-inf")
    with duckdb.connect() as db:
        query = _feature_query(features_dir, split, label_before)
        for batch in db.execute(query).fetch_record_batch(100_000):
            frame = batch.to_pandas()
            actual = frame.target_demand_next_day.to_numpy(float)
            predictions = {name: predict_baseline(frame, name) for name in BASELINES}
            predictions["lightgbm"] = model.predict(frame)
            prediction_min = min(prediction_min, predictions["lightgbm"].min())
            prediction_max = max(prediction_max, predictions["lightgbm"].max())
            for name, predicted in predictions.items():
                batches.append({
                    "method": name, "rows": len(actual),
                    "absolute": mean_absolute_error(actual, predicted) * len(actual),
                    "squared": mean_squared_error(actual, predicted) * len(actual),
                })
    if not batches:
        raise ValueError("Cannot evaluate an empty split")
    # Sum batch errors first; averaging batch MAEs would bias short final batches.
    totals = pd.DataFrame(batches).groupby("method").sum()
    metrics = pd.DataFrame({
        "rows": totals.rows, "mae": totals.absolute / totals.rows,
        "rmse": np.sqrt(totals.squared / totals.rows),
    })
    return metrics.to_dict(orient="index"), bool(prediction_max - prediction_min > 1e-9)


def run_pipeline(
    features_dir=ROOT / "data/features/forecasting",
    output_dir=ROOT / "data/production/forecasting",
    model_path=ROOT / "models/forecast_model.pkl",
    tracking_uri=TRACKING_URI,
    max_train_rows=1_000_000,
    n_estimators=100,
    seed=42,
):
    """Prepare data, fit LightGBM, compare baselines, and save the result."""
    # 1. Prepare data.
    train, validation, weights, metadata, population = prepare_data(
        features_dir, max_train_rows, seed
    )

    # 2. Train the model.
    model = train_model(train, validation, weights, n_estimators, seed)

    # 3. Evaluate against baselines on validation and test.
    ranges = metadata["date_ranges"]
    test_start = pd.Timestamp(ranges["test"]["min"])
    validation_metrics, varying = evaluate_model(model, features_dir, "validation", test_start)
    test_metrics, _ = evaluate_model(model, features_dir, "test")
    eligible_baselines = [name for name in BASELINES if name != "always_zero"]
    baseline = min(eligible_baselines, key=lambda name: (validation_metrics[name]["mae"], name))
    report = {
        "candidate": "lightgbm",
        "baseline": baseline,
        "validation": validation_metrics,
        "test": test_metrics,
        "validation_approved": forecast_gate(
            validation_metrics["lightgbm"], validation_metrics[baseline], varying
        ),
        "holdout_passed": forecast_gate(
            test_metrics["lightgbm"], test_metrics[baseline]
        ),
    }

    # 4. Save results, log to MLflow, and publish only if both checks pass.
    params = {
        "target": "target_demand_next_day",
        "training_rows": len(train),
        "training_population_rows": population,
        "max_train_rows": max_train_rows,
        "seed": seed,
        "n_estimators": n_estimators,
        "trained_estimators": model.estimator.n_estimators_,
        "validation_range": json.dumps(ranges["validation"]),
        "test_range": json.dumps(ranges["test"]),
    }

    return save_run(
        model, report, params, output_dir, model_path, tracking_uri,
        experiment="ecommerce-demand-forecasting",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=ROOT / "data/features/forecasting")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/production/forecasting")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/forecast_model.pkl")
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_pipeline(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
