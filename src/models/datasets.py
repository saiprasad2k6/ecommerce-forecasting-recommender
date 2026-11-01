"""Shared input validation, data loading and feature preparation for baseline and candidate models."""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "demand", "demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_28",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_28",
]
READ_COLUMNS = [
    "purchase_date", "product_id", "customer_state", *NUMERIC_FEATURES,
    "demand_lag_6", "demand_lag_27", "target_demand_next_day",
]
RANK_FEATURES = [
    "global_recent", "global_all_time", "regional_recent", "category_affinity",
    "category_popularity", "customer_frequency", "days_since_purchase", "product_age_days",
]


def validate_inputs(features_dir, warehouse, candidate_limit=100, top_k=10):
    if not 1 <= top_k <= candidate_limit:
        raise ValueError("Require 1 <= top-k <= candidate-limit")
    metadata = json.loads((features_dir / "metadata.json").read_text())
    ranges = []
    with duckdb.connect() as db:
        for split in ["train", "validation", "test"]:
            path = features_dir / f"{split}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; run olist_data_preparation first")
            table = db.read_parquet(str(path))
            columns = set(table.columns)
            missing = set(READ_COLUMNS) - columns
            if missing:
                raise ValueError(
                    f"{path} is missing columns: {sorted(missing)}; rebuild forecasting features"
                )
            minimum, maximum, count = table.aggregate(
                "MIN(purchase_date), MAX(purchase_date), COUNT(*)"
            ).fetchone()
            if not count:
                raise ValueError(f"Empty {split} split")
            actual = {
                "min": pd.Timestamp(minimum).strftime("%Y-%m-%d"),
                "max": pd.Timestamp(maximum).strftime("%Y-%m-%d"),
            }
            if actual != metadata["date_ranges"][split] or count != metadata["rows"][split]:
                raise ValueError(f"{split} data does not match metadata.json; rebuild features")
            ranges.append((minimum, maximum))
    if ranges[0][1] >= ranges[1][0] or ranges[1][1] >= ranges[2][0]:
        raise ValueError("Train/validation/test periods overlap")
    if not warehouse.is_file():
        raise FileNotFoundError(warehouse)
    try:
        with duckdb.connect(str(warehouse), read_only=True) as db:
            db.execute("""SELECT order_id, order_item_id, customer_unique_id, product_id,
                customer_state, product_category_name_english, order_status,
                order_purchase_timestamp, order_delivered_customer_date
                FROM order_items LIMIT 1""")
    except duckdb.IOException as error:
        raise RuntimeError(
            "Cannot open warehouse. Disconnect DBeaver and wait for warehouse-writing tasks "
            "to finish."
        ) from error
    return {"status": "ready", "features_dir": str(features_dir), "warehouse": str(warehouse)}


def feature_query(features_dir, split, label_before=None):
    """Read prepared features, excluding labels from the next period when needed."""
    path = str(features_dir / f"{split}.parquet").replace("'", "''")
    query = f"""
        SELECT {', '.join(READ_COLUMNS)}
        FROM read_parquet('{path}')
    """
    if label_before is not None:
        query += f"WHERE purchase_date + INTERVAL 1 DAY < DATE '{label_before.date()}'"
    return query

def load_history(warehouse: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Load delivered purchase history strictly before the snapshot cutoff date."""
    with duckdb.connect(str(warehouse), read_only=True) as con:
        return con.execute("""
            SELECT order_id, order_item_id, customer_unique_id, product_id,
                   customer_state,
                   COALESCE(product_category_name_english, 'unknown') AS category,
                   CAST(order_purchase_timestamp AS TIMESTAMP) AS purchased_at
            FROM order_items
            WHERE order_status = 'delivered'
              AND CAST(order_purchase_timestamp AS TIMESTAMP) < ?
              AND CAST(order_delivered_customer_date AS TIMESTAMP) < ?
        """, [cutoff.to_pydatetime(), cutoff.to_pydatetime()]).df()

def load_labels(warehouse, start, end, label_before=None):
    """Read purchase labels whose delivery outcome was known before the boundary."""
    label_before = end if label_before is None else label_before
    with duckdb.connect(str(warehouse), read_only=True) as db:
        return db.execute("""
            SELECT DISTINCT customer_unique_id, product_id FROM order_items
            WHERE order_status = 'delivered'
              AND CAST(order_purchase_timestamp AS TIMESTAMP) >= ?
              AND CAST(order_purchase_timestamp AS TIMESTAMP) < ?
              AND CAST(order_delivered_customer_date AS TIMESTAMP) < ?
            ORDER BY customer_unique_id, product_id
        """, [start.to_pydatetime(), end.to_pydatetime(), label_before.to_pydatetime()]).df()

class RecommendationSnapshot:
    """Past-only product and customer statistics used by every recommender."""

    def __init__(self, history, cutoff, candidate_limit=100):
        self.cutoff = pd.Timestamp(cutoff)
        self.candidate_limit = candidate_limit
        if history.empty or candidate_limit < 1:
            raise ValueError("A snapshot needs history and a positive candidate limit")
        if history.purchased_at.max() >= self.cutoff:
            raise ValueError("Snapshot history contains purchases at/after its cutoff")
        recent = history.loc[history.purchased_at >= self.cutoff - pd.Timedelta(days=90)]

        self.products = history.groupby("product_id").agg(
            category=("category", "first"), global_all_time=("order_id", "size"),
            first_purchase=("purchased_at", "min"),
        )
        self.products["global_recent"] = recent.product_id.value_counts().reindex(self.products.index, fill_value=0)
        self.products["category_popularity"] = self.products.category.map(recent.category.value_counts()).fillna(0)
        self.products["product_age_days"] = (self.cutoff - self.products.pop("first_purchase")) / pd.Timedelta(days=1)
        popularity = self.products.global_recent if not recent.empty else self.products.global_all_time
        popularity = popularity.loc[popularity > 0].sort_index().sort_values(ascending=False, kind="stable")
        self.global_order = popularity.index.tolist()
        self.popular_categories = self.products.category.reindex(self.global_order)
        self.catalog = set(self.products.index)

        purchases = history.groupby(["customer_unique_id", "product_id"], as_index=False).agg(
            customer_frequency=("order_id", "size"), last_purchase=("purchased_at", "max"),
        )
        purchases = purchases.sort_values(
            ["customer_unique_id", "customer_frequency", "last_purchase", "product_id"],
            ascending=[True, False, False, True],
        )
        self.customer_products = purchases.groupby("customer_unique_id").product_id.agg(list).to_dict()
        self.purchases = purchases.set_index(["customer_unique_id", "product_id"])
        latest = history.sort_values("purchased_at", kind="stable").drop_duplicates("customer_unique_id", keep="last")
        self.states = latest.set_index("customer_unique_id").customer_state
        self.regional_counts = recent.groupby(["customer_state", "product_id"]).size()
        self.category_affinity = recent.groupby("customer_unique_id").category.value_counts(normalize=True)
        self.customer_categories = recent.groupby("customer_unique_id").category.agg(set).to_dict()

    def prepare_features(self, customer):
        categories = self.customer_categories.get(customer, set())
        matching_products = self.popular_categories.loc[self.popular_categories.isin(categories)]
        candidates = (self.global_order[:self.candidate_limit]
                      + matching_products.index[:self.candidate_limit].tolist()
                      + self.customer_products.get(customer, [])[:self.candidate_limit])
        product_ids = pd.Index(candidates, name="product_id").drop_duplicates()
        features = self.products.loc[product_ids].copy()

        purchase_index = pd.MultiIndex.from_product([[customer], product_ids])
        purchases = self.purchases.reindex(purchase_index).droplevel(0)
        empty = pd.Series(dtype=float)
        state = self.states.get(customer, "unknown")
        features["regional_recent"] = self.regional_counts.get(state, empty).reindex(product_ids, fill_value=0)
        features["category_affinity"] = features.category.map(self.category_affinity.get(customer, empty)).fillna(0)
        features["customer_frequency"] = purchases.customer_frequency.fillna(0)
        features["days_since_purchase"] = ((self.cutoff - purchases.last_purchase) / pd.Timedelta(days=1)).fillna(-1)
        count_columns = ["global_recent", "global_all_time", "regional_recent", "category_popularity", "customer_frequency"]
        features[count_columns] = np.log1p(features[count_columns])
        return features[RANK_FEATURES].astype("float32")
