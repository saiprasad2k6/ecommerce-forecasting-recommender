"""Replay boundaries keep August outcomes out of development datasets."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.common.config import REPLAY
from src.models.forecasting.forecasting_baseline import evaluate_baselines_duckdb
from src.features.build_forecasting_features import build_features, split_features, write_outputs


class ForecastingReplayTests(unittest.TestCase):
    def test_target_date_boundaries_and_unlabeled_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daily = pd.DataFrame({
                "purchase_date": pd.date_range("2018-04-01", "2018-08-05"),
                "product_id": "P1", "customer_state": "SP", "order_item_count": 2,
            })
            for date, frame in daily.groupby("purchase_date"):
                folder = root / f"purchase_date={date.date()}"
                folder.mkdir()
                frame.to_csv(folder / "daily_demand.csv", index=False)
            features = build_features(root)
            boundaries = {key: REPLAY[key] for key in
                          ["validation_start", "test_start", "monitoring_start", "monitoring_end"]}
            splits = split_features(features, **boundaries)
            for name, first, last in [
                ("validation", "2018-06-01", "2018-06-30"),
                ("test", "2018-07-01", "2018-07-31"),
            ]:
                targets = splits[name].purchase_date + pd.Timedelta(days=1)
                self.assertEqual(targets.min(), pd.Timestamp(first))
                self.assertEqual(targets.max(), pd.Timestamp(last))
            self.assertLess((splits["train"].purchase_date + pd.Timedelta(days=1)).max(), pd.Timestamp("2018-06-01"))
            self.assertEqual(splits["inference"].purchase_date.max(), pd.Timestamp("2018-08-05"))
            self.assertNotIn("target_demand_next_day", splits["inference"].columns)
            self.assertTrue(splits["test"].target_demand_next_day.notna().all())
            metadata = write_outputs(splits, root / "output", **boundaries)
            self.assertEqual(metadata["boundaries"], boundaries)
            self.assertTrue((root / "output/inference.parquet").is_file())
            metrics = evaluate_baselines_duckdb(root / "output/validation.parquet", "validation")
            self.assertTrue(metrics.rows.eq(30).all())
            # Products first seen later must not create earlier product/state rows.
            later = root / "purchase_date=2018-08-06"
            later.mkdir()
            pd.DataFrame({"purchase_date": ["2018-08-06"], "product_id": ["NEW"],
                          "customer_state": ["SP"], "order_item_count": [1]}).to_csv(
                later / "daily_demand.csv", index=False)
            updated = build_features(root)
            self.assertFalse(updated.product_id.eq("NEW").any())
            before = features.loc[features.purchase_date.lt("2018-08-05")].reset_index(drop=True)
            after = updated.loc[updated.purchase_date.lt("2018-08-05")].reset_index(drop=True)
            pd.testing.assert_frame_equal(before, after)

    def test_rejects_reversed_boundaries(self):
        with self.assertRaisesRegex(ValueError, "chronological"):
            split_features(pd.DataFrame(), "2018-07-01", "2018-06-01", "2018-08-01", "2018-09-01")
