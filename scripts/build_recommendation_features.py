#!/usr/bin/env python3
"""Build purchase-based recommendation candidates and a weighted baseline.

Reads the warehouse produced by prepare_curated_data.py. Only purchases delivered
before --as-of are used. No availability, views, brands or discounts are inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RECENT_DAYS = 90
WEIGHTS = {
    "category_affinity": 0.4,
    "regional_popularity_score": 0.3,
    "global_popularity_score": 0.2,
    "co_purchase_score": 0.1,
}


def load_history(warehouse: Path, as_of: pd.Timestamp) -> pd.DataFrame:
    if not warehouse.is_file():
        raise FileNotFoundError(warehouse)
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        return connection.execute("""
            SELECT order_id, order_item_id, customer_unique_id, product_id,
                   customer_state,
                   COALESCE(product_category_name_english, 'unknown') AS category,
                     CAST(order_purchase_timestamp AS TIMESTAMP) AS purchased_at
            FROM order_items
            WHERE order_status = 'delivered'
              AND CAST(order_purchase_timestamp AS TIMESTAMP) < ?
              AND CAST(order_delivered_customer_date AS TIMESTAMP) < ?
        """, [as_of.to_pydatetime(), as_of.to_pydatetime()]).df()


def top_products(frame: pd.DataFrame, column: str, limit: int) -> list[str]:
    return (
        frame.sort_values(column, ascending=False, kind="stable")
        .head(limit)
        .index.tolist()
    )


def add_sources(
    sources: dict[str, set[str]], products: list[str], source: str
) -> None:
    for product_id in products:
        sources.setdefault(product_id, set()).add(source)


def normalize(values: pd.Series, maximum: float) -> pd.Series:
    return values / maximum if maximum > 0 else 0.0


def build_candidates(
    history: pd.DataFrame,
    customer_id: str,
    state: str,
    as_of: pd.Timestamp,
    candidate_limit: int = 50,
    exclude_purchased: bool = True,
) -> pd.DataFrame:
    """Return at most four candidate_limit-sized source lists, deduplicated.

    Category matches are a basic similarity proxy, not learned item similarity.
    Historical median price is descriptive; it is not a current catalog price.
    """
    if candidate_limit <= 0:
        raise ValueError("candidate-limit must be positive")
    if history.empty:
        raise ValueError("No delivered purchases are known before the cutoff")
    mine = history.loc[history.customer_unique_id.eq(customer_id)]
    recent = history.loc[
        history.purchased_at.ge(as_of - pd.Timedelta(days=RECENT_DAYS))
    ]
    recent_mine = mine.loc[
        mine.purchased_at.ge(as_of - pd.Timedelta(days=RECENT_DAYS))
    ]
    catalog = history.groupby("product_id").agg(category=("category", "first"))
    global_counts = recent.groupby("product_id").size()
    regional_counts = recent.loc[recent.customer_state.eq(state)].groupby("product_id").size()
    catalog["global_purchases_90d"] = global_counts.reindex(catalog.index, fill_value=0)
    catalog["regional_purchases_90d"] = regional_counts.reindex(catalog.index, fill_value=0)
    catalog["all_time_purchases"] = history.groupby("product_id").size()

    # Count distinct shared baskets, avoiding repeated items within an order.
    basket_items = history[["order_id", "product_id"]].drop_duplicates()
    seed_orders = basket_items.loc[basket_items.product_id.isin(mine.product_id), "order_id"]
    co_counts = basket_items.loc[basket_items.order_id.isin(seed_orders)].groupby("product_id").size()
    catalog["co_purchase_count"] = co_counts.reindex(catalog.index, fill_value=0)
    affinities = recent_mine.category.value_counts(normalize=True)
    catalog["category_affinity"] = catalog.category.map(affinities).fillna(0.0)
    if exclude_purchased:
        catalog = catalog.loc[~catalog.index.isin(mine.product_id)].copy()

    sources: dict[str, set[str]] = {}
    popularity_column = (
        "global_purchases_90d"
        if catalog.global_purchases_90d.any()
        else "all_time_purchases"
    )
    add_sources(sources, top_products(catalog, popularity_column, candidate_limit), "global_popular")
    add_sources(
        sources,
        top_products(
            catalog.loc[catalog.regional_purchases_90d.gt(0)],
            "regional_purchases_90d",
            candidate_limit,
        ),
        "regional_popular",
    )
    add_sources(
        sources,
        top_products(catalog.loc[catalog.category_affinity.gt(0)], popularity_column, candidate_limit),
        "category_match",
    )
    add_sources(
        sources,
        top_products(catalog.loc[catalog.co_purchase_count.gt(0)], "co_purchase_count", candidate_limit),
        "bought_together",
    )
    result = catalog.loc[list(sources)].copy()
    result["candidate_sources"] = [",".join(sorted(sources[p])) for p in result.index]
    result["customer_unique_id"] = customer_id
    result["customer_state"] = state
    result["as_of"] = as_of
    result["customer_segment"] = (
        "new"
        if mine.empty
        else "repeat"
        if mine.order_id.nunique() > 1
        else "one_order"
    )
    score_columns = {
        "regional_popularity_score": "regional_purchases_90d",
        "global_popularity_score": popularity_column,
        "co_purchase_score": "co_purchase_count",
    }
    for output, column in score_columns.items():
        result[output] = normalize(result[column], catalog[column].max())

    result["baseline_score"] = sum(
        result[column] * weight for column, weight in WEIGHTS.items()
    )
    if mine.empty:
        result["baseline_score"] = (
            0.6 * result.regional_popularity_score
            + 0.4 * result.global_popularity_score
        )
    result = result.rename_axis("product_id").reset_index()
    result = result.sort_values(["baseline_score", "product_id"], ascending=[False, True])
    result["rank"] = range(1, len(result) + 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", type=Path, default=ROOT / "data/warehouse/olist.duckdb")
    parser.add_argument("--customer-id", required=True, help="customer_unique_id, not customer_id")
    parser.add_argument("--state", required=True)
    parser.add_argument("--as-of", required=True, help="Exclusive cutoff, e.g. 2018-08-01")
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-purchased", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/features/recommendations")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("top-k must be positive")
    as_of = pd.Timestamp(args.as_of)
    if pd.isna(as_of) or as_of.tzinfo is not None:
        parser.error("as-of must be a valid timezone-naive timestamp matching warehouse timestamps")
    history = load_history(args.warehouse, as_of)
    candidates = build_candidates(history, args.customer_id, args.state, as_of,
                                  args.candidate_limit, not args.include_purchased)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(args.output_dir / "candidates.parquet", index=False)
    candidates.head(args.top_k).to_parquet(args.output_dir / "recommendations.parquet", index=False)
    metadata = {"as_of": str(as_of), "customer_unique_id": args.customer_id,
                "candidate_count": len(candidates), "weights": WEIGHTS,
                "availability_checked": False,
                "limitations": "Historical purchases only; no live inventory, recalls, views, brands or promotions."}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(candidates.head(args.top_k).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
