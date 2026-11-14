"""Persisted prediction loading, enrichment, idempotency and transactional rollback."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import pandas as pd

from src.inference import prediction_store as prepare
from src.versioning import snapshots


class PredictionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        patcher = patch.object(snapshots, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.references = self.root / "reference"
        self.references.mkdir()
        self.predictions = self.root / "predictions"
        self.database = self.root / "predictions.duckdb"
        self.run_id = "manual__2026-11-14T13:00:00+00:00"
        pd.DataFrame([
            {"product_id": "p1", "product_category_name": "casa", "product_weight_g": 200,
             "product_length_cm": 10, "product_height_cm": 2, "product_width_cm": 5, "product_photos_qty": 1},
        ]).to_csv(self.references / "products.csv", index=False)
        pd.DataFrame([{"product_category_name": "casa", "product_category_name_english": "home"}]).to_csv(
            self.references / "categories.csv", index=False)
        pd.DataFrame([
            {"customer_id": "old", "customer_unique_id": "customer1", "customer_state": "SP", "customer_city": "sao paulo"},
            {"customer_id": "future", "customer_unique_id": "customer1", "customer_state": "RJ", "customer_city": "rio"},
        ]).to_csv(self.references / "customers.csv", index=False)
        pd.DataFrame([
            {"order_id": "a", "customer_id": "old", "order_purchase_timestamp": "2018-07-01"},
            {"order_id": "b", "customer_id": "future", "order_purchase_timestamp": "2018-08-02"},
        ]).to_csv(self.references / "orders.csv", index=False)
        self.write_batch()

    def write_batch(self, run_id=None, suffix="batch"):
        common = {"as_of": pd.Timestamp("2018-08-01"), "model_name": "model",
                  "model_role": "candidate", "training_pipeline_run_id": "training-1",
                  "pipeline_run_id": run_id or self.run_id}
        self.forecast = pd.DataFrame([
            {**common, "model_run_id": "forecast-model", "product_id": "p1", "customer_state": "SP",
             "purchase_date": pd.Timestamp("2018-08-01"), "target_date": pd.Timestamp("2018-08-02"), "prediction": 3.5},
            {**common, "model_run_id": "forecast-model", "product_id": "missing", "customer_state": "ZZ",
             "purchase_date": pd.Timestamp("2018-08-01"), "target_date": pd.Timestamp("2018-08-02"), "prediction": 1.0},
        ])
        self.recommendation = pd.DataFrame([
            {**common, "model_run_id": "rec-model", "customer_unique_id": "customer1", "recommended_products": ["missing", "p1"], "k": 2},
            {**common, "model_run_id": "rec-model", "customer_unique_id": "unknown", "recommended_products": ["p1"], "k": 2},
        ])
        self.paths = {}
        for task, frame in (("forecasting", self.forecast), ("recommendation", self.recommendation)):
            path = self.predictions / task / suffix / "predictions.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            self.paths[task] = path

    def build(self):
        return prepare.run_pipeline(self.predictions, self.references, self.database)

    def query(self, sql, parameters=None):
        with duckdb.connect(str(self.database), read_only=True) as connection:
            return connection.execute(sql, parameters or []).df()

    def test_load_enrich_and_query_without_original_parquet(self):
        report = self.build()
        self.assertEqual(report["loaded"][0]["row_counts"]["recommendation_items"], 3)
        forecasts = self.query("SELECT * FROM forecasts ORDER BY prediction DESC")
        self.assertEqual(forecasts.category.tolist(), ["home", "Unknown"])
        self.assertEqual(forecasts.state_name.tolist(), ["São Paulo", "Unknown"])
        items = self.query("SELECT * FROM recommendations WHERE customer_unique_id='customer1' ORDER BY rank")
        self.assertEqual(items.product_id.tolist(), ["missing", "p1"])
        self.assertEqual(items['rank'].tolist(), [1, 2])
        self.assertEqual(items.customer_state.tolist(), ["SP", "SP"])
        for path in self.paths.values():
            path.unlink()
        for path in self.references.glob('*.csv'):
            path.unlink()
        self.assertEqual(len(self.query("SELECT * FROM forecasts")), 2)
        self.assertEqual(len(self.query("SELECT * FROM recommendations")), 3)
        self.assertEqual(self.query("SELECT sum(predicted_demand) AS total FROM forecast_summary").total.iloc[0], 4.5)

    def test_reimport_does_not_duplicate_rows_or_change_original_enrichment(self):
        self.build()
        references = pd.read_csv(self.references / 'categories.csv')
        references['product_category_name_english'] = 'changed'
        references.to_csv(self.references / 'categories.csv', index=False)
        self.assertEqual(self.build()['loaded'][0]['status'], 'already_loaded')
        self.assertEqual(len(self.query('SELECT * FROM forecast_predictions')), 2)
        self.assertEqual(self.query("SELECT category FROM forecasts WHERE product_id='p1'").category.iloc[0], 'home')
        self.write_batch('manual__2026-11-15T13:00:00+00:00', suffix='new')
        self.build()
        self.assertEqual(len(self.query('SELECT * FROM prediction_runs')), 2)
        self.assertEqual(len(self.query('SELECT * FROM forecast_predictions')), 4)
        self.assertEqual(set(self.query("SELECT category FROM forecasts WHERE product_id='p1'").category), {'home','changed'})

    def test_incomplete_backfill_is_skipped_but_requested_batch_fails(self):
        self.paths['recommendation'].unlink()
        self.assertEqual(len(self.build()['skipped_incomplete']), 1)
        self.assertEqual(len(self.query('SELECT * FROM prediction_runs')), 0)
        with self.assertRaisesRegex(ValueError, 'Both prediction tasks'):
            prepare.run_pipeline(self.predictions, self.references, self.database, batch_run_id=self.run_id)
        self.assertEqual(len(self.query('SELECT * FROM prediction_runs')), 0)

    def test_invalid_prediction_rolls_back_all_tables(self):
        self.forecast.loc[0, 'prediction'] = -1
        self.forecast.to_parquet(self.paths['forecasting'], index=False)
        with self.assertRaisesRegex(ValueError, 'Invalid forecasting'):
            self.build()
        self.assertEqual(len(self.query('SHOW TABLES')), 0)

    def test_changed_predictions_for_loaded_run_are_rejected(self):
        self.build()
        self.forecast.loc[0, 'prediction'] = 99
        self.forecast.to_parquet(self.paths['forecasting'], index=False)
        with self.assertRaisesRegex(ValueError, 'Prediction bytes changed'):
            self.build()
        self.assertEqual(self.query('SELECT sum(prediction) AS total FROM forecasts').total.iloc[0], 4.5)

    def test_duplicate_reference_keys_cannot_multiply_predictions(self):
        path = self.references / 'products.csv'
        products = pd.read_csv(path)
        pd.concat([products,products]).to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError, 'Duplicate reference keys'):
            self.build()
        self.assertEqual(len(self.query('SHOW TABLES')), 0)

    def test_source_changes_during_load_are_not_committed(self):
        original = prepare.load_predictions
        def mutate(connection, batch):
            report = original(connection, batch)
            self.forecast.loc[0,'prediction'] = 99
            self.forecast.to_parquet(self.paths['forecasting'],index=False)
            return report
        with patch.object(prepare,'load_predictions',side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed during loading'):
                self.build()
        self.assertEqual(len(self.query('SHOW TABLES')),0)

    def test_prediction_storage_is_materialized_but_enrichment_is_not_duplicated(self):
        self.build()
        kinds = dict(self.query('SELECT table_name, table_type FROM information_schema.tables').itertuples(index=False,name=None))
        for name in ('forecast_predictions','recommendation_predictions','prediction_runs'):
            self.assertEqual(kinds[name], 'BASE TABLE')
        for name in ('forecasts','recommendations','forecast_summary','recommendation_summary'):
            self.assertEqual(kinds[name], 'VIEW')

    def test_generation_timestamp_is_preserved(self):
        for task, frame in (('forecasting',self.forecast),('recommendation',self.recommendation)):
            frame['generated_at'] = '2026-11-15T10:00:00+00:00'
            frame.to_parquet(self.paths[task],index=False)
        self.build()
        self.assertTrue(self.query('SELECT run_timestamp FROM prediction_runs').run_timestamp.iloc[0].startswith('2026-11-15'))
