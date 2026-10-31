"""Shared evaluation data and metrics for independent baseline/candidate jobs."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
REC_PERIODS = {
    "validation": (pd.Timestamp("2018-07-01"), pd.Timestamp("2018-08-01")),
    "test": (pd.Timestamp("2018-08-01"), pd.Timestamp("2018-09-01")),
}

def sql_path(path):
    return str(path).replace("'", "''")

def feature_query(features_dir, split, label_before=None):
    """Read prepared features, excluding labels from the next period when needed."""
    path = sql_path(features_dir / f"{split}.parquet")
    query = f"""
        SELECT {', '.join(READ_COLUMNS)}
        FROM read_parquet('{path}')
    """
    if label_before is not None:
        query += f"WHERE purchase_date + INTERVAL 1 DAY < DATE '{label_before.date()}'"
    return query

def load_history(warehouse: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Load delivered purchase history strictly before the snapshot cutoff date."""
    if not warehouse.is_file():
        raise FileNotFoundError(f"Missing DuckDB warehouse: {warehouse}")
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
    """Candidate retrieval and features computed once from past-only history.

    Repeat products are eligible for every method. Candidates are the union of
    global, category and purchase-history lists; future labels are never injected.
    """

    def __init__(self, history, cutoff, candidate_limit=100):
        if history.empty or candidate_limit < 1:
            raise ValueError("A snapshot needs purchase history and a positive candidate limit")
        cutoff = pd.Timestamp(cutoff)
        if history.purchased_at.max() >= cutoff:
            raise ValueError("Snapshot history contains purchases at/after its cutoff")
        self.cutoff = cutoff
        self.candidate_limit = candidate_limit

        # Product popularity and category indexes, shared by all customers.
        recent = history.loc[history.purchased_at.ge(cutoff - pd.Timedelta(days=90))]
        counts = recent.groupby("product_id").size()
        all_counts = history.groupby("product_id").size()
        popularity = counts if not counts.empty else all_counts
        popularity = popularity.rename("count").reset_index()
        popularity = popularity.sort_values(["count", "product_id"], ascending=[False, True])
        self.global_order = popularity.product_id.tolist()
        self.catalog = set(all_counts.index)
        products = history.drop_duplicates("product_id").set_index("product_id")
        self.categories = products.category
        self.popular_categories = self.categories.reindex(self.global_order)
        self.global_counts = counts
        self.all_counts = all_counts
        self.regional = recent.groupby(["customer_state", "product_id"]).size()
        self.category_counts = recent.groupby("category").size()
        self.first = history.groupby("product_id").purchased_at.min()

        # Customer history: last known state, product frequency and recency.
        latest = history.sort_values("purchased_at", kind="stable")
        latest = latest.drop_duplicates("customer_unique_id", keep="last")
        self.states = latest.set_index("customer_unique_id").customer_state.to_dict()
        preferences = history.groupby(["customer_unique_id", "product_id"], as_index=False).agg(
            count=("order_id", "size"), last=("purchased_at", "max")
        )
        indexed_preferences = preferences.set_index(["customer_unique_id", "product_id"])
        self.frequency = indexed_preferences["count"]
        self.last = indexed_preferences["last"]
        preferences = preferences.sort_values(
            ["customer_unique_id", "count", "last", "product_id"],
            ascending=[True, False, False, True],
        )
        self.customer_products = (
            preferences.groupby("customer_unique_id").product_id.agg(list).to_dict()
        )

        # A customer's category affinity is their share of recent purchases there.
        self.affinity = recent.groupby("customer_unique_id").category.value_counts(normalize=True)
        self.customer_categories = recent.groupby("customer_unique_id").category.agg(set).to_dict()

    def prepare_features(self, customer):
        limit = self.candidate_limit
        popular = self.global_order[:limit]
        categories = self.customer_categories.get(customer, set())
        matches = self.popular_categories.loc[self.popular_categories.isin(categories)]
        category_matches = matches.index[:limit].tolist()
        previously_purchased = self.customer_products.get(customer, [])[:limit]
        ids = pd.Index(popular + category_matches + previously_purchased, name="product_id")
        ids = ids.drop_duplicates()
        state = self.states.get(customer, "unknown")
        empty = pd.Series(dtype=float)
        frame = pd.DataFrame(index=ids)
        product_categories = ids.map(self.categories)
        frame["global_recent"] = self.global_counts.reindex(ids, fill_value=0)
        frame["global_all_time"] = self.all_counts.reindex(ids, fill_value=0)
        frame["regional_recent"] = self.regional.get(state, empty).reindex(ids, fill_value=0)
        frame["category_affinity"] = product_categories.map(
            self.affinity.get(customer, empty)
        ).fillna(0)
        frame["category_popularity"] = product_categories.map(self.category_counts).fillna(0)
        frame["customer_frequency"] = self.frequency.get(customer, empty).reindex(ids, fill_value=0)
        last_purchase = pd.to_datetime(self.last.get(customer, empty).reindex(ids))
        frame["days_since_purchase"] = (
            (self.cutoff - last_purchase).dt.total_seconds() / 86400
        ).fillna(-1)
        frame["product_age_days"] = (self.cutoff - self.first.reindex(ids)).dt.total_seconds() / 86400
        counts = ["global_recent", "global_all_time", "regional_recent",
                  "category_popularity", "customer_frequency"]
        frame[counts] = np.log1p(frame[counts])
        return frame[RANK_FEATURES].astype("float32")

def ranking_metrics(recommendations: list[str], relevant: set[str], k: int) -> dict[str, float]:
    """Compute the fraction of recommendations and purchases matched at K."""
    if k < 1:
        raise ValueError("k must be positive")
    hits = len(set(recommendations[:k]) & relevant)
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(relevant) if relevant else 0.0,
    }


def evaluate_forecasting(models, features_dir, split, label_before=None):
    """Score only the supplied models on the identical next-day rows."""
    batches, ranges = [], {name: [float("inf"), float("-inf")] for name in models}
    with duckdb.connect() as db:
        for batch in db.execute(feature_query(features_dir, split, label_before)).fetch_record_batch(100_000):
            frame = batch.to_pandas()
            actual = frame.target_demand_next_day.to_numpy(float)
            for name, model in models.items():
                predicted = np.asarray(model.predict(frame), dtype=float)
                ranges[name][0] = min(ranges[name][0], predicted.min())
                ranges[name][1] = max(ranges[name][1], predicted.max())
                batches.append({
                    "method": name, "rows": len(actual),
                    "absolute": mean_absolute_error(actual, predicted) * len(actual),
                    "squared": mean_squared_error(actual, predicted) * len(actual),
                })
    if not batches:
        raise ValueError("Cannot evaluate an empty split")
    totals = pd.DataFrame(batches).groupby("method").sum()
    metrics = pd.DataFrame({
        "rows": totals.rows, "mae": totals.absolute / totals.rows,
        "rmse": np.sqrt(totals.squared / totals.rows),
    }).to_dict(orient="index")
    varying = {name: bool(high - low > 1e-9) for name, (low, high) in ranges.items()}
    return metrics, varying


def evaluate_recommendations(models, snapshot, labels, k):
    """Score only the supplied models, including misses outside their candidates."""
    truths = labels.groupby("customer_unique_id").product_id.agg(set).to_dict()
    if not truths:
        raise ValueError("No delivered purchase labels in evaluation period")
    records = []
    for customer, relevant in truths.items():
        for name, model in models.items():
            products = model.predict(snapshot, customer, k)
            records.append({"method": name, **ranking_metrics(products, relevant, k)})
    metrics = pd.DataFrame(records).groupby("method").mean().to_dict(orient="index")
    for row in metrics.values():
        row["customers"] = len(truths)
    return metrics
