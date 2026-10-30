"""Train a purchase-based LambdaRank recommender using historical snapshots."""

import argparse
import json
from pathlib import Path

import duckdb
from lightgbm import LGBMRanker
import numpy as np
import pandas as pd

from src.models.baseline.recommendation_baselines import load_history, ranking_metrics
from src.models.production.common import (
    ROOT, TRACKING_URI, ranking_gate, save_run,
)


BASELINES = ["global_popularity", "category_popularity", "purchase_history"]
NUMERIC_FEATURES = [
    "global_recent", "global_all_time", "regional_recent", "category_affinity",
    "category_popularity", "customer_frequency", "days_since_purchase", "product_age_days",
]


def predict_baseline(snapshot, method, customer, k=10):
    """Predict from historical popularity, category preference or purchase history."""
    if method == "global_popularity":
        selected = snapshot.global_order[:k]
    elif method == "category_popularity":
        categories = snapshot.customer_categories.get(customer, set())
        matches = snapshot.popular_categories.loc[snapshot.popular_categories.isin(categories)]
        selected = matches.index[:k].tolist() + snapshot.global_order[:k]
    elif method == "purchase_history":
        selected = snapshot.customer_products.get(customer, [])[:k] + snapshot.global_order[:k]
    else:
        raise ValueError(method)
    return pd.Index(selected).drop_duplicates()[:k].tolist()


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
        return frame[NUMERIC_FEATURES].astype("float32")


class ProductionRecommender:
    """Use a validated ranker, otherwise the selected baseline retrieval rule."""

    def __init__(self, estimator=None, fallback="global_popularity", use_model=True):
        self.estimator = estimator
        self.fallback = fallback
        self.use_model = use_model

    def predict(self, snapshot, customer, k=10):
        if not 1 <= k <= snapshot.candidate_limit:
            raise ValueError("k must be between 1 and the snapshot candidate limit")
        if not self.use_model:
            return predict_baseline(snapshot, self.fallback, customer, k)
        frame = snapshot.prepare_features(customer)
        ranked = pd.DataFrame({"product_id": frame.index, "score": self.estimator.predict(frame)})
        ranked = ranked.sort_values(["score", "product_id"], ascending=[False, True])
        return ranked.product_id.head(k).tolist()

    recommend = predict  # Keep existing saved-model examples working.


def _load_labels(warehouse, start, end, label_before):
    """Read purchase labels whose delivery outcome was known before the boundary."""
    with duckdb.connect(str(warehouse), read_only=True) as db:
        return db.execute("""
            SELECT DISTINCT customer_unique_id, product_id FROM order_items
            WHERE order_status = 'delivered'
              AND CAST(order_purchase_timestamp AS TIMESTAMP) >= ?
              AND CAST(order_purchase_timestamp AS TIMESTAMP) < ?
              AND CAST(order_delivered_customer_date AS TIMESTAMP) < ?
            ORDER BY customer_unique_id, product_id
        """, [start.to_pydatetime(), end.to_pydatetime(), label_before.to_pydatetime()]).df()


def prepare_data(warehouse, start, validation_start, candidate_limit=100):
    """For each month, use earlier history to predict that month's purchases."""
    frames, targets, groups = [], [], []
    boundaries = pd.date_range(start, validation_start, freq=pd.DateOffset(months=1))
    boundaries = boundaries.union(pd.DatetimeIndex([validation_start]))
    for cutoff, label_end in zip(boundaries[:-1], boundaries[1:]):
        history = load_history(warehouse, cutoff)
        snapshot = RecommendationSnapshot(history, cutoff, candidate_limit)
        labels = _load_labels(warehouse, cutoff, label_end, validation_start)
        for customer, relevant in labels.groupby("customer_unique_id").product_id.agg(set).items():
            frame = snapshot.prepare_features(customer)
            target = frame.index.isin(relevant).astype("int32")
            # A ranking group needs both purchased and unpurchased candidates.
            if target.any() and not target.all():
                frames.append(frame)
                targets.extend(target)
                groups.append(len(frame))
    if not frames:
        raise ValueError(
            "No informative ranking groups; increase candidates or extend training dates"
        )
    return pd.concat(frames, ignore_index=True), np.asarray(targets), groups


def train_model(train, targets, groups, n_estimators=100, seed=42):
    """Fit a ranker; each group contains one customer's candidate products."""
    if n_estimators < 1:
        raise ValueError("n-estimators must be positive")
    estimator = LGBMRanker(
        objective="lambdarank", metric="None", n_estimators=n_estimators,
        random_state=seed, n_jobs=2, verbosity=-1,
    )
    estimator.fit(train, targets, group=groups)
    return ProductionRecommender(estimator)


def evaluate_model(model, snapshot, labels, k):
    """Compare the ranker and three baselines over the same customers."""
    truths = labels.groupby("customer_unique_id").product_id.agg(set).to_dict()
    if not truths:
        raise ValueError("No delivered purchase labels in evaluation period")
    records = []
    for customer, relevant in truths.items():
        methods = {name: predict_baseline(snapshot, name, customer, k) for name in BASELINES}
        methods["lightgbm_ranker"] = model.predict(snapshot, customer, k)
        for method, products in methods.items():
            records.append({
                "method": method,
                **ranking_metrics(products, relevant, k),
            })
    frame = pd.DataFrame(records)
    metrics = frame.groupby("method").mean(numeric_only=True).to_dict(orient="index")
    for row in metrics.values():
        row["customers"] = len(truths)
    return metrics


def run_pipeline(
    warehouse=ROOT / "data/warehouse/olist.duckdb",
    output_dir=ROOT / "data/production/recommendations",
    model_path=ROOT / "models/recommender.pkl",
    tracking_uri=TRACKING_URI,
    training_start="2018-03-01",
    validation_cutoff="2018-07-01",
    test_cutoff="2018-08-01",
    test_end="2018-09-01",
    candidate_limit=100,
    top_k=10,
    n_estimators=100,
    seed=42,
):
    """Prepare monthly examples, fit the ranker, evaluate, and save the result."""
    # 1. Prepare data.
    start = pd.Timestamp(training_start)
    validation_start = pd.Timestamp(validation_cutoff)
    test_start = pd.Timestamp(test_cutoff)
    end = pd.Timestamp(test_end)
    dates = [start, validation_start, test_start, end]
    if any(date.tzinfo is not None or pd.isna(date) for date in dates):
        raise ValueError("Use timezone-naive dates matching warehouse timestamps")
    if not start < validation_start < test_start < end:
        raise ValueError("Training, validation, test cutoffs and test end must be chronological")
    if not 1 <= top_k <= candidate_limit:
        raise ValueError("Require 1 <= top-k <= candidate-limit")
    train, targets, groups = prepare_data(
        warehouse, start, validation_start, candidate_limit
    )

    # 2. Train the model.
    model = train_model(train, targets, groups, n_estimators, seed)

    # 3. Evaluate against baselines on validation and test.
    validation_snapshot = RecommendationSnapshot(
        load_history(warehouse, validation_start), validation_start, candidate_limit
    )
    validation_labels = _load_labels(warehouse, validation_start, test_start, test_start)

    validation_metrics = evaluate_model(
        model, validation_snapshot, validation_labels, top_k
    )
    baseline = min(BASELINES, key=lambda name: (
        -validation_metrics[name]["recall_at_k"],
        -validation_metrics[name]["precision_at_k"], name,
    ))
    test_snapshot = RecommendationSnapshot(
        load_history(warehouse, test_start), test_start, candidate_limit
    )
    test_labels = _load_labels(warehouse, test_start, end, end)
    test_metrics = evaluate_model(model, test_snapshot, test_labels, top_k)
    report = {
        "candidate": "lightgbm_ranker",
        "baseline": baseline,
        "validation": validation_metrics,
        "test": test_metrics,
        "validation_approved": ranking_gate(
            validation_metrics["lightgbm_ranker"], validation_metrics[baseline]
        ),
        "holdout_passed": ranking_gate(
            test_metrics["lightgbm_ranker"], test_metrics[baseline]
        ),
    }

    # 4. Save results, log to MLflow, and publish only if both checks pass.
    params = {
        "seed": seed,
        "n_estimators": n_estimators,
        "trained_estimators": model.estimator.n_estimators_,
        "training_start": str(start.date()),
        "training_rows": len(train),
        "training_groups": len(groups),
        "validation_range": f"{validation_start.date()} to {test_start.date()}",
        "test_range": f"{test_start.date()} to {end.date()}",
        "k": top_k,
        "candidate_limit": candidate_limit,
    }
    return save_run(
        model, report, params, output_dir, model_path, tracking_uri,
        experiment="ecommerce-recommendations",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", type=Path, default=ROOT / "data/warehouse/olist.duckdb")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/production/recommendations")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/recommender.pkl")
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--training-start", default="2018-03-01")
    parser.add_argument("--validation-cutoff", default="2018-07-01")
    parser.add_argument("--test-cutoff", default="2018-08-01")
    parser.add_argument("--test-end", default="2018-09-01")
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_pipeline(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
