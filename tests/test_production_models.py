"""Temporal alignment, retrieval, fallback and real training/serialization checks."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.comparison import forecast_gate, ranking_gate
from src.models.production import common, forecasting, recommendation
from src.models.production.forecasting import ProductionForecaster
from src.models.production.recommendation import (
    RecommendationSnapshot,
)


@contextmanager
def isolated_tracking(root):
    previous = Path.cwd()
    os.chdir(root)
    try:
        with patch.object(common, "ROOT", root):
            yield f"sqlite:///{root / 'mlflow.db'}"
    finally:
        os.chdir(previous)


def forecast_fixture(root):
    frames = []
    for product in range(2):
        frame = pd.DataFrame({"purchase_date": pd.date_range("2018-01-01", periods=120),
                              "product_id": f"p{product}", "customer_state": "SP"})
        frame["demand"] = np.arange(120) % 7 + product
        for lag in [1, 6, 7, 14, 27, 28]:
            frame[f"demand_lag_{lag}"] = frame.demand.shift(lag)
        frame["rolling_mean_7"] = frame.demand.shift(1).rolling(7).mean()
        frame["rolling_std_7"] = frame.demand.shift(1).rolling(7).std()
        frame["rolling_mean_28"] = frame.demand.shift(1).rolling(28).mean()
        frame["target_demand_next_day"] = frame.demand.shift(-1)
        frames.append(frame.dropna())
    frame = pd.concat(frames, ignore_index=True)
    dates = sorted(frame.purchase_date.unique())
    split_frames = {"train": frame.loc[frame.purchase_date.lt(dates[50])],
                    "validation": frame.loc[frame.purchase_date.ge(dates[50]) & frame.purchase_date.lt(dates[70])],
                    "test": frame.loc[frame.purchase_date.ge(dates[70])]}
    root.mkdir(parents=True)
    metadata = {"rows": {}, "date_ranges": {}}
    for name, data in split_frames.items():
        data.to_parquet(root / f"{name}.parquet", index=False)
        metadata["rows"][name] = len(data)
        metadata["date_ranges"][name] = {"min": data.purchase_date.min().strftime("%Y-%m-%d"),
                                         "max": data.purchase_date.max().strftime("%Y-%m-%d")}
    (root / "metadata.json").write_text(json.dumps(metadata))
    return frame, metadata


def warehouse_fixture(path):
    rows = []
    for month in range(2, 9):
        for customer in range(20):
            purchased = pd.Timestamp(2018, month, 5)
            rows.append({"order_id": f"o{month}_{customer}", "order_item_id": 1,
                "customer_unique_id": f"c{customer}", "product_id": f"p{customer % 5}",
                "customer_state": "SP", "product_category_name_english": "sports",
                "order_status": "delivered", "order_purchase_timestamp": purchased,
                "order_delivered_customer_date": purchased + pd.Timedelta(days=2)})
    frame = pd.DataFrame(rows)
    with duckdb.connect(str(path)) as db:
        db.register("fixture", frame.astype(object))
        db.execute("CREATE TABLE order_items AS SELECT * FROM fixture")


class ProductionModelTests(unittest.TestCase):
    def test_gates_reject_rmse_and_holdout_regressions(self):
        self.assertFalse(forecast_gate({"mae": 0.5, "rmse": 2},
                                             {"mae": 1, "rmse": 1}))
        self.assertFalse(forecast_gate({"mae": 0.5, "rmse": 0.5},
                                             {"mae": 1, "rmse": 1}, varying=False))
        baseline = {"precision_at_k": 0.3, "recall_at_k": 0.4}
        validation = ranking_gate({"precision_at_k": 0.4, "recall_at_k": 0.5}, baseline)
        holdout = ranking_gate({"precision_at_k": 0.4, "recall_at_k": 0.2}, baseline)
        self.assertTrue(validation)
        self.assertFalse(validation and holdout)

    def test_next_day_alignment_and_weighted_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "features"
            frame, metadata = forecast_fixture(root)
            sample = frame.loc[frame.product_id.eq("p0")].tail(10)
            # The synthetic series repeats weekly: the correctly aligned weekly
            # prediction must equal tomorrow's label, not today's lag_7.
            np.testing.assert_array_equal(forecasting.predict_baseline(sample, "same_day_last_week"), sample.target_demand_next_day)
            np.testing.assert_allclose(forecasting.predict_baseline(sample, "moving_average_7"), 3)
            start = pd.Timestamp(metadata["date_ranges"]["validation"]["min"])
            sampled, _, weights, _, population = forecasting.prepare_data(root, 40, 42)
            self.assertLessEqual(len(sampled), 40)
            self.assertTrue(((sampled.purchase_date + pd.Timedelta(days=1)) < start).all())
            self.assertAlmostEqual(weights.sum(), population)

            # Prepared lags must keep their meaning when reading a split directly.
            test_start = pd.Timestamp(metadata["date_ranges"]["test"]["min"])
            weekly_model = ProductionForecaster(use_model=False, fallback="same_day_last_week")
            metrics, _ = forecasting.evaluate_model(weekly_model, root, "validation", test_start)
            expected_rows = frame.loc[
                frame.purchase_date.ge(start)
                & (frame.purchase_date + pd.Timedelta(days=1)).lt(test_start)
            ].shape[0]
            self.assertEqual(metrics["lightgbm"]["rows"], expected_rows)
            self.assertEqual(metrics["lightgbm"]["mae"], 0)

            # Older feature files get a clear rebuild instruction at preflight.
            test_path = root / "test.parquet"
            pd.read_parquet(test_path).drop(columns="demand_lag_6").to_parquet(test_path)
            from src.models.production.validate_inputs import validate_inputs
            with self.assertRaisesRegex(ValueError, "rebuild forecasting features"):
                validate_inputs(root, root / "unused.duckdb")

    def test_rejected_candidate_preserves_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            common.publish_model({"previous": True}, path, True)
            self.assertEqual(common.publish_model({"candidate": True}, path, False), "retained_existing_model")
            self.assertEqual(joblib.load(path), {"previous": True})

    def test_forecast_metrics_match_full_data_with_an_unequal_last_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = np.arange(100_003)
            frame = pd.DataFrame({
                "purchase_date": pd.Timestamp("2018-01-01") + pd.to_timedelta(row % 60, unit="D"),
                "product_id": (row // 60).astype(str), "customer_state": "SP",
                "demand": row % 7, "target_demand_next_day": row % 5,
            })
            for name in ["demand_lag_1", "demand_lag_6", "demand_lag_7", "demand_lag_14",
                         "demand_lag_27", "demand_lag_28", "rolling_mean_7",
                         "rolling_std_7", "rolling_mean_28"]:
                frame[name] = 0.0
            frame.to_parquet(root / "test.parquet", index=False)
            model = ProductionForecaster(use_model=False, fallback="last_day")
            metrics, _ = forecasting.evaluate_model(model, root, "test")
            actual, predicted = frame.target_demand_next_day, frame.demand
            self.assertEqual(metrics["lightgbm"]["rows"], len(frame))
            self.assertAlmostEqual(metrics["lightgbm"]["mae"],
                                   mean_absolute_error(actual, predicted))
            self.assertAlmostEqual(metrics["lightgbm"]["rmse"],
                                   mean_squared_error(actual, predicted) ** 0.5)
            self.assertEqual(set(metrics["lightgbm"]), {"rows", "mae", "rmse"})

            frame["target_demand_next_day"] = 0
            frame.to_parquet(root / "test.parquet", index=False)
            zero_metrics, _ = forecasting.evaluate_model(model, root, "test")
            self.assertAlmostEqual(zero_metrics["lightgbm"]["mae"], predicted.mean())
            self.assertTrue(np.isfinite(zero_metrics["lightgbm"]["rmse"]))

    def test_forecasting_training_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame, _ = forecast_fixture(root / "features")
            with isolated_tracking(root) as tracking:
                report = forecasting.run_pipeline(root / "features", root / "outputs", tracking, max_train_rows=100, n_estimators=5)
            self.assertIn("mlflow_run_id", report)
            client = common.mlflow.MlflowClient(tracking_uri=tracking)
            run = client.get_run(report["mlflow_run_id"])
            self.assertEqual(run.info.status, "FINISHED")
            self.assertAlmostEqual(run.data.metrics["val_mae"],
                                   report["validation"]["mae"])
            self.assertEqual(set(run.data.metrics), {"val_mae", "val_rmse", "test_mae", "test_rmse"})
            candidate = joblib.load(Path(report["output_dir"]) / "model.pkl")
            self.assertIsInstance(candidate, ProductionForecaster)
            predicted = candidate.predict(frame.head(10))
            self.assertTrue(np.isfinite(predicted).all())
            self.assertTrue((predicted >= 0).all())
            self.assertFalse((root / "model.pkl").exists())
            unknown = frame.head(1).copy()
            unknown["product_id"] = "unknown-product"
            self.assertEqual(candidate.prepare_features(unknown).product_id_code.iloc[0], -1)

    def test_logging_failure_cannot_publish_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecast_fixture(root / "features")
            with isolated_tracking(root) as tracking:
                with patch.object(common.mlflow, "log_artifact",
                                  side_effect=RuntimeError("tracking unavailable")):
                    with self.assertRaisesRegex(RuntimeError, "tracking unavailable"):
                        forecasting.run_pipeline(root / "features", root / "outputs", tracking, max_train_rows=100, n_estimators=5)
            self.assertFalse((root / "model.pkl").exists())

    def test_baseline_pipelines_log_only_basic_metrics(self):
        from src.models.baseline.forecasting_baselines import run_baseline_pipeline
        from src.models.baseline.recommendation_baselines import run_recommendation_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecast_fixture(root / "features")
            warehouse_fixture(root / "olist.duckdb")
            with isolated_tracking(root) as tracking:
                forecasting = run_baseline_pipeline(root / "features", root / "forecast", tracking)
                recommendations = run_recommendation_pipeline(
                    root / "olist.duckdb", root / "recommendations", tracking, k=3
                )
            client = common.mlflow.MlflowClient(tracking_uri=tracking)
            for result in forecasting["runs"]:
                metrics = client.get_run(result["mlflow_run_id"]).data.metrics
                self.assertEqual(set(metrics), {"val_mae", "val_rmse", "test_mae", "test_rmse"})
            for result in recommendations["runs"]:
                metrics = client.get_run(result["mlflow_run_id"]).data.metrics
                self.assertEqual(set(metrics), {
                    "val_precision_at_k", "val_recall_at_k",
                    "test_precision_at_k", "test_recall_at_k",
                })

    def test_recommendation_training_retrieval_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warehouse_fixture(root / "olist.duckdb")
            with isolated_tracking(root) as tracking:
                report = recommendation.run_pipeline(root / "olist.duckdb", root / "outputs", tracking, candidate_limit=5, top_k=3, n_estimators=5)
            from src.models.baseline.recommendation_baselines import load_history
            cutoff = pd.Timestamp("2018-08-01")
            snapshot = RecommendationSnapshot(load_history(root / "olist.duckdb", cutoff), cutoff, 5)
            features = snapshot.prepare_features("c1").loc["p1"]
            self.assertAlmostEqual(features.global_recent, np.log1p(12), places=6)
            self.assertAlmostEqual(features.global_all_time, np.log1p(24), places=6)
            self.assertAlmostEqual(features.customer_frequency, np.log1p(6), places=6)
            self.assertEqual(features.category_affinity, 1)
            self.assertEqual(features.days_since_purchase, 27)
            cold = snapshot.prepare_features("new-customer")
            self.assertTrue(cold.customer_frequency.eq(0).all())
            self.assertTrue(cold.category_affinity.eq(0).all())
            self.assertTrue(cold.regional_recent.eq(0).all())
            self.assertTrue(cold.days_since_purchase.eq(-1).all())
            self.assertFalse((root / "model.pkl").exists())
            model = joblib.load(Path(report["output_dir"]) / "model.pkl")
            products = model.predict(snapshot, "c1", 3)
            self.assertEqual(len(products), len(set(products)))
            self.assertTrue(set(products).issubset(snapshot.catalog))
            unknown = model.predict(snapshot, "new-customer", 3)
            self.assertEqual(len(unknown), 3)
            self.assertFalse("future-product" in snapshot.prepare_features("c1").index)
            self.assertEqual(len(recommendation.predict_baseline(snapshot, "category_popularity", "c1", 1)), 1)
            self.assertEqual(len(recommendation.predict_baseline(snapshot, "category_popularity", "c1", 3)), 3)
            labels = recommendation._load_labels(root / "olist.duckdb", pd.Timestamp("2018-07-01"), cutoff, pd.Timestamp("2018-07-06"))
            self.assertTrue(labels.empty, "Labels delivered after known_by must not be used")
            self.assertIn("mlflow_run_id", report)
            client = common.mlflow.MlflowClient(tracking_uri=tracking)
            run = client.get_run(report["mlflow_run_id"])
            self.assertEqual(set(run.data.metrics), {
                "val_precision_at_k", "val_recall_at_k",
                "test_precision_at_k", "test_recall_at_k",
            })


if __name__ == "__main__":
    unittest.main()
