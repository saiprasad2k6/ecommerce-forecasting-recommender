"""Only activated model bundles can serve; inference does not choose algorithms."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.inference.batch import load_active_model, run_pipeline
from src.models.common import publish_model
from src.models.forecasting.forecasting_baseline import BaselineForecaster
from src.models.recommendation.recommendation_baseline import BaselineRecommender
from test_production_models import forecast_fixture, warehouse_fixture


class BatchInferenceTests(unittest.TestCase):
    def selection(self, task, name, role="candidate"):
        return {"task": task, "approved_for_inference": True, "model_name": name,
                "model_role": role, "model_run_id": name + "_run",
                "training_pipeline_run_id": "training_001", "k": 3, "candidate_limit": 5}

    def test_switches_between_approved_algorithms_and_keeps_rejected_model_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecast_fixture(root / "features")
            source = root / "features/test.parquet"
            features = pd.read_parquet(source).drop(columns="target_demand_next_day")
            features.to_parquet(source, index=False)
            date = features.purchase_date.min()
            active = root / "models/forecast_model.pkl"
            publish_model(BaselineForecaster("last_day"), active, False,
                          metadata=self.selection("forecasting", "last_day", "baseline"))
            first = run_pipeline("forecasting", date, source, root / "models", root / "predictions", "inference_001")
            initial = pd.read_parquet(first["output"])
            expected = pd.read_parquet(source).query("purchase_date == @date").sort_values(["product_id", "customer_state"])
            np.testing.assert_array_equal(initial.prediction, expected.demand)
            self.assertTrue(initial.model_name.eq("last_day").all())
            self.assertTrue(initial.pipeline_run_id.eq("inference_001").all())

            another_algorithm = DummyRegressor(strategy="constant", constant=9).fit([[0], [1]], [0, 1])
            publish_model(another_algorithm, active, True, metadata=self.selection("forecasting", "another_algorithm"))
            with patch("mlflow.search_runs", side_effect=AssertionError("Inference must not search training runs")):
                second = run_pipeline("forecasting", date, source, root / "models", root / "predictions")
            self.assertTrue(pd.read_parquet(second["output"]).prediction.eq(9).all())
            self.assertNotEqual(first["output"], second["output"])
            self.assertTrue(Path(first["output"]).exists())

            publish_model(BaselineForecaster("always_zero"), active, False,
                          metadata=self.selection("forecasting", "rejected"))
            selected = load_active_model("forecasting", root / "models")
            self.assertEqual(selected.serving_metadata["model_name"], "another_algorithm")
            again = run_pipeline("forecasting", date, source, root / "models", root / "predictions")
            self.assertEqual(again["output"], second["output"])
            self.assertTrue(pd.read_parquet(again["output"]).prediction.eq(9).all())

    def test_requires_serving_approval_and_matching_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "No active"):
                load_active_model("forecasting", root)
            path = root / "forecast_model.pkl"
            joblib.dump(BaselineForecaster(), path)
            with self.assertRaisesRegex(ValueError, "serving approval"):
                load_active_model("forecasting", root)
            publish_model(BaselineForecaster(), path, True,
                          metadata=self.selection("recommendation", "wrong_task"))
            with self.assertRaisesRegex(ValueError, "serving approval"):
                load_active_model("forecasting", root)

    def test_recommendations_use_the_selected_baseline_and_its_k(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warehouse_fixture(root / "olist.duckdb")
            publish_model(BaselineRecommender("purchase_history"), root / "models/recommender.pkl", False,
                          metadata=self.selection("recommendation", "purchase_history", "baseline"))
            result = run_pipeline("recommendation", "2018-08-01", root / "olist.duckdb",
                                  root / "models", root / "predictions", "inference_001")
            predictions = pd.read_parquet(result["output"])
            self.assertEqual(len(predictions), 20)
            self.assertTrue(predictions.k.eq(3).all())
            self.assertTrue(predictions.model_role.eq("baseline").all())
            self.assertTrue(predictions.recommended_products.map(len).eq(3).all())
            products = predictions.set_index("customer_unique_id").loc["c1", "recommended_products"]
            self.assertEqual(products[0], "p1")


if __name__ == "__main__":
    unittest.main()
