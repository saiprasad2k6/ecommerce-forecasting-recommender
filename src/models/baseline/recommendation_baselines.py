#!/usr/bin/env python3
"""Product Recommendation Baseline Models & MLflow Tracking Pipeline.

Implements three classical e-commerce recommendation baseline strategies,
offline ranking metrics (Precision@K, Recall@K, HitRate@K, NDCG@K),
cold-start cohort breakdown (Warm vs. New customers), and MLflow tracking
under the dedicated experiment 'ecommerce-recommendations'.

Available baselines:
  - global_popularity: Most purchased products across the platform in the last 90 days.
  - category_popularity: Top products from the customer's recent categories, fallback to global popularity.
  - purchase_history: Customer's previously purchased products (frequency + recency), fallback to global popularity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import duckdb
import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
WAREHOUSE = ROOT / "data" / "warehouse" / "olist.duckdb"
BASELINES = ROOT / "data" / "baselines"
DEFAULT_TRACKING_URI = f"sqlite:///{BASELINES / 'mlflow.db'}"
DEFAULT_EXPERIMENT_NAME = "ecommerce-recommendations"
RECENT_DAYS = 90
DEFAULT_K = 10

REC_PERIODS = {
    "validation": (pd.Timestamp("2018-07-01"), pd.Timestamp("2018-08-01")),
    "test": (pd.Timestamp("2018-08-01"), pd.Timestamp("2018-09-01")),
}


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


def load_labels(warehouse: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load ground truth purchases made during the evaluation period."""
    if not warehouse.is_file():
        raise FileNotFoundError(f"Missing DuckDB warehouse: {warehouse}")
    with duckdb.connect(str(warehouse), read_only=True) as db:
        return db.execute("""
            SELECT DISTINCT customer_unique_id, product_id
            FROM order_items
            WHERE order_status = 'delivered'
              AND CAST(order_purchase_timestamp AS TIMESTAMP) >= ?
              AND CAST(order_purchase_timestamp AS TIMESTAMP) < ?
            ORDER BY customer_unique_id, product_id
        """, [start.to_pydatetime(), end.to_pydatetime()]).df()


def unique_top(products: list[str], k: int) -> list[str]:
    """Deduplicate while preserving order, truncated to size k."""
    return list(dict.fromkeys(products))[:k]


def prepare_recommender(history: pd.DataFrame, cutoff: pd.Timestamp):
    """Precompute indexed candidate lists for global, category, and user purchase history."""
    recent = history.loc[history.purchased_at.ge(cutoff - pd.Timedelta(days=RECENT_DAYS))]
    popular_history = recent if not recent.empty else history

    counts = popular_history.groupby("product_id").size().rename("count").reset_index()
    global_order = counts.sort_values(["count", "product_id"], ascending=[False, True]).product_id.tolist()

    categories = history.drop_duplicates("product_id").set_index("product_id").category.to_dict()
    recent_categories = recent.groupby("customer_unique_id").category.agg(set).to_dict()

    preferences = history.groupby(["customer_unique_id", "product_id"], as_index=False).agg(
        frequency=("order_id", "size"),
        last_purchase=("purchased_at", "max"),
    )
    preferences = preferences.sort_values(
        ["customer_unique_id", "frequency", "last_purchase", "product_id"],
        ascending=[True, False, False, True],
    )
    history_orders = preferences.groupby("customer_unique_id").product_id.agg(list).to_dict()

    category_lists: dict[frozenset[str], list[str]] = {}

    def recommend(customer: str, method: str, k: int) -> list[str]:
        if method == "global_popularity":
            selected = global_order
        elif method == "category_popularity":
            key = frozenset(recent_categories.get(customer, set()))
            if key not in category_lists:
                category_lists[key] = [p for p in global_order if categories.get(p) in key][:k]
            selected = category_lists[key] + global_order[:k]
        elif method == "purchase_history":
            selected = history_orders.get(customer, [])[:k] + global_order[:k]
        else:
            raise ValueError(f"Unknown recommendation method: '{method}'")
        return unique_top(selected, k)

    return recommend


def ranking_metrics(recommendations: list[str], relevant: set[str], k: int) -> dict[str, float]:
    """Compute Precision@K, Recall@K, HitRate@K, and NDCG@K."""
    recommendations = recommendations[:k]
    hits = [int(p in relevant) for p in recommendations]
    dcg = sum(hit / math.log2(position + 2) for position, hit in enumerate(hits))
    ideal = sum(1 / math.log2(position + 2) for position in range(min(k, len(relevant))))
    return {
        "precision_at_k": sum(hits) / k,
        "recall_at_k": sum(hits) / len(relevant) if relevant else 0.0,
        "hit_rate_at_k": float(any(hits)),
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
    }


class BaselineRecommender:
    """Standardized baseline recommender interface for top-K product ranking."""

    def __init__(
        self,
        history: pd.DataFrame,
        cutoff: pd.Timestamp,
        method: str = "purchase_history",
        k: int = DEFAULT_K,
    ) -> None:
        self.method = method
        self.k = k
        self.cutoff = cutoff
        self._recommender_fn = prepare_recommender(history, cutoff)

    def recommend(self, customer_id: str, k: int | None = None) -> list[str]:
        top_k = k or self.k
        return self._recommender_fn(customer_id, self.method, top_k)


def run_recommendation_pipeline(
    warehouse_path: Path = WAREHOUSE,
    output_dir: Path = BASELINES,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Run offline evaluation and log runs to MLflow under 'ecommerce-recommendations'."""
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = ["global_popularity", "category_popularity", "purchase_history"]

    recommendation_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    example_recommendations: list[dict[str, Any]] = []

    for split, (cutoff, end) in REC_PERIODS.items():
        history = load_history(warehouse_path, cutoff)
        labels = load_labels(warehouse_path, cutoff, end)
        recommend = prepare_recommender(history, cutoff)

        truths = labels.groupby("customer_unique_id").product_id.agg(set).to_dict()
        known_customers = set(history.customer_unique_id)
        catalog_ids = set(history.product_id)
        label_catalog_coverage = float(labels.product_id.isin(catalog_ids).mean())

        for method in methods:
            records: list[dict[str, Any]] = []
            recommended_catalog: set[str] = set()

            for customer, relevant in truths.items():
                products = recommend(customer, method, k)
                recommended_catalog.update(products)
                is_warm = customer in known_customers

                scores = ranking_metrics(products, relevant, k)
                scores["cohort"] = "warm" if is_warm else "new"
                records.append(scores)

                if customer == next(iter(truths)):
                    example_recommendations.append({
                        "split": split,
                        "baseline": method,
                        "customer_id": customer[:8] + "...",
                        "recommendations": ", ".join([p[:8] + "..." for p in products[:5]]),
                        "actual_purchases": ", ".join([p[:8] + "..." for p in sorted(relevant)]),
                    })

            scores_df = pd.DataFrame(records)
            overall = scores_df.drop(columns="cohort").mean().to_dict()

            recommendation_rows.append({
                "split": split,
                "baseline": method,
                **overall,
                "customers": len(scores_df),
                "warm_customers": int(scores_df.cohort.eq("warm").sum()),
                "catalog_coverage": len(recommended_catalog) / len(catalog_ids) if catalog_ids else 0.0,
                "label_catalog_coverage": label_catalog_coverage,
            })

            for cohort, group in scores_df.groupby("cohort"):
                cohort_rows.append({
                    "split": split,
                    "baseline": method,
                    "cohort": cohort,
                    "customers": len(group),
                    **group.drop(columns="cohort").mean().to_dict(),
                })

    metrics_df = pd.DataFrame(recommendation_rows)
    cohorts_df = pd.DataFrame(cohort_rows)
    example_df = pd.DataFrame(example_recommendations)

    metrics_csv = output_dir / "recommendation_metrics.csv"
    cohorts_csv = output_dir / "recommendation_cohorts.csv"
    example_csv = output_dir / "example_recommendations.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    cohorts_df.to_csv(cohorts_csv, index=False)
    example_df.to_csv(example_csv, index=False)

    # Champion selection on validation NDCG@K
    val_subset = metrics_df.query("split == 'validation'")
    champion_baseline = val_subset.sort_values("ndcg_at_k", ascending=False).iloc[0]["baseline"]

    # MLflow Tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logged_runs: list[dict[str, Any]] = []
    for method in methods:
        is_champion = method == champion_baseline
        with mlflow.start_run(run_name=f"baseline__{method}") as run:
            # Parameters
            mlflow.log_params({
                "task": "recommendation",
                "model_family": "heuristic_baseline",
                "baseline_name": method,
                "k": k,
                "popularity_window_days": RECENT_DAYS,
                "repeat_products_allowed": True,
                "inventory_filter": "unavailable_in_olist",
                "validation_snapshot": str(REC_PERIODS["validation"][0].date()),
                "validation_label_end": str(REC_PERIODS["validation"][1].date()),
                "test_snapshot": str(REC_PERIODS["test"][0].date()),
                "test_label_end": str(REC_PERIODS["test"][1].date()),
                "is_champion_baseline": is_champion,
            })

            # Tags
            mlflow.set_tag("pipeline_integrated", "true")
            mlflow.set_tag("selected_on_validation", str(is_champion))
            if is_champion:
                mlflow.set_tag("model_status", "champion_baseline")

            # Metrics
            for split in ["validation", "test"]:
                split_prefix = "val" if split == "validation" else "test"
                row = metrics_df.loc[(metrics_df["baseline"] == method) & (metrics_df["split"] == split)].iloc[0]

                mlflow.log_metrics({
                    f"{split_prefix}_ndcg_at_k": float(row["ndcg_at_k"]),
                    f"{split_prefix}_recall_at_k": float(row["recall_at_k"]),
                    f"{split_prefix}_precision_at_k": float(row["precision_at_k"]),
                    f"{split_prefix}_hit_rate_at_k": float(row["hit_rate_at_k"]),
                    f"{split_prefix}_catalog_coverage": float(row["catalog_coverage"]),
                    f"{split_prefix}_customers": int(row["customers"]),
                    f"{split_prefix}_warm_customers": int(row["warm_customers"]),
                })

                for cohort in ["warm", "new"]:
                    cohort_match = cohorts_df.loc[
                        (cohorts_df["baseline"] == method)
                        & (cohorts_df["split"] == split)
                        & (cohorts_df["cohort"] == cohort)
                    ]
                    if not cohort_match.empty:
                        c_row = cohort_match.iloc[0]
                        mlflow.log_metrics({
                            f"{split_prefix}_{cohort}_ndcg_at_k": float(c_row["ndcg_at_k"]),
                            f"{split_prefix}_{cohort}_recall_at_k": float(c_row["recall_at_k"]),
                        })

            # Artifacts
            mlflow.log_artifact(str(metrics_csv))
            mlflow.log_artifact(str(cohorts_csv))
            mlflow.log_artifact(str(example_csv))

            logged_runs.append({
                "baseline": method,
                "run_id": run.info.run_id,
                "val_ndcg_at_k": float(val_subset.loc[val_subset["baseline"] == method, "ndcg_at_k"].iloc[0]),
                "val_recall_at_k": float(val_subset.loc[val_subset["baseline"] == method, "recall_at_k"].iloc[0]),
                "champion": is_champion,
            })

    return {
        "status": "success",
        "champion_baseline": champion_baseline,
        "champion_val_ndcg": float(val_subset.loc[val_subset["baseline"] == champion_baseline, "ndcg_at_k"].iloc[0]),
        "champion_val_recall": float(val_subset.loc[val_subset["baseline"] == champion_baseline, "recall_at_k"].iloc[0]),
        "runs": logged_runs,
        "metrics_csv": str(metrics_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", type=Path, default=WAREHOUSE, help="DuckDB warehouse path")
    parser.add_argument("--output-dir", type=Path, default=BASELINES, help="Artifacts output directory")
    parser.add_argument("--tracking-uri", type=str, default=DEFAULT_TRACKING_URI, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT_NAME, help="MLflow experiment name")
    parser.add_argument("--top-k", type=int, default=DEFAULT_K, help="Cutoff K for ranking metrics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 70)
    print("Running Product Recommendation Baseline Evaluation Pipeline")
    print(f"Warehouse:  {args.warehouse}")
    print(f"Output:     {args.output_dir}")
    print(f"Tracking:   {args.tracking_uri}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Top K:      {args.top_k}")
    print("=" * 70)

    results = run_recommendation_pipeline(
        warehouse_path=args.warehouse,
        output_dir=args.output_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        k=args.top_k,
    )

    print("\nLeaderboard Summary:")
    runs_df = pd.DataFrame(results["runs"]).sort_values("val_ndcg_at_k", ascending=False).reset_index(drop=True)
    print(runs_df.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"🏆 CHAMPION BASELINE: '{results['champion_baseline']}'")
    print(f"   Validation NDCG@{args.top_k}: {results['champion_val_ndcg']:.6f}")
    print(f"   Validation Recall@{args.top_k}: {results['champion_val_recall']:.6f}")
    print("=" * 70)
    print(f"Summary metrics saved to: {results['metrics_csv']}")
    print(f"Tracked in MLflow experiment: {args.experiment_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
