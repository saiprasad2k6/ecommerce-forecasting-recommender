"""Behavior checks for cutoff handling, candidate generation and fallback."""
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from src.features.build_recommendation_features import build_candidates, load_history


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = pd.Timestamp("2018-08-01")
        self.history = pd.DataFrame([
            ("o1", "c1", "shoe", "SP", "sports", 100),
            ("o2", "c2", "shoe", "SP", "sports", 100),
            ("o2", "c2", "sock", "SP", "sports", 20),
            ("o3", "c3", "mixer", "RJ", "kitchen", 200),
        ], columns=["order_id", "customer_unique_id", "product_id",
                    "customer_state", "category", "price"])
        self.history["purchased_at"] = pd.Timestamp("2018-07-01")

    def test_personalized_candidates_and_exclusion(self):
        result = build_candidates(self.history, "c1", "SP", self.cutoff)
        self.assertNotIn("shoe", result.product_id.tolist())
        self.assertEqual(result.iloc[0].product_id, "sock")
        self.assertIn("bought_together", result.iloc[0].candidate_sources)
        self.assertEqual(result.iloc[0].category_affinity, 1)
        self.assertEqual(result.product_id.nunique(), len(result))

    def test_new_customer_and_empty_recent_window(self):
        result = build_candidates(self.history, "new", "XX", pd.Timestamp("2019-08-01"), 1)
        self.assertEqual(result.iloc[0].product_id, "shoe")
        self.assertEqual(result.iloc[0].customer_segment, "new")
        self.assertGreater(result.iloc[0].baseline_score, 0)

    def test_cutoff_excludes_future_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.duckdb"
            frame = self.history.rename(columns={"category": "product_category_name_english",
                                                  "purchased_at": "order_purchase_timestamp"}).copy()
            frame["order_item_id"] = range(len(frame))
            frame["order_status"] = "delivered"
            frame["order_delivered_customer_date"] = pd.Timestamp("2018-07-10")
            frame.loc[frame.product_id.eq("mixer"), "order_delivered_customer_date"] = self.cutoff
            with duckdb.connect(str(path)) as connection:
                connection.register("fixture", frame.astype(object))
                connection.execute("CREATE TABLE order_items AS SELECT * FROM fixture")
            result = load_history(path, self.cutoff)
            self.assertNotIn("mixer", result.product_id.tolist())


if __name__ == "__main__":
    unittest.main()
