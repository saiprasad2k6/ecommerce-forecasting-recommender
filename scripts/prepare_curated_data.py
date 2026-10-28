#!/usr/bin/env python3
"""Build curated Olist tables from data/processed without changing raw data."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CURATED = ROOT / "data" / "curated"
WAREHOUSE = ROOT / "data" / "warehouse" / "olist.duckdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROCESSED)
    parser.add_argument("--output-dir", type=Path, default=CURATED)
    parser.add_argument("--warehouse", type=Path, default=WAREHOUSE)
    parser.add_argument("--purchase-date", help="Rebuild one YYYY-MM-DD partition")
    return parser.parse_args()


def read_table(input_dir: Path, name: str, dates: list[str] | None = None) -> pd.DataFrame:
    path = input_dir / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cleaned table: {path}")
    return pd.read_csv(path, parse_dates=dates or [])


def build_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    customers = read_table(input_dir, "customers")
    orders = read_table(
        input_dir,
        "orders",
        [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    items = read_table(input_dir, "order_items", ["shipping_limit_date"])
    products = read_table(input_dir, "products")
    sellers = read_table(input_dir, "sellers")
    payments = read_table(input_dir, "payments")
    reviews = read_table(input_dir, "reviews")
    categories = read_table(input_dir, "categories")

    payment_totals = payments.groupby("order_id", as_index=False).agg(
        payment_value_total=("payment_value", "sum"),
        payment_count=("payment_sequential", "count"),
    )
    review_totals = reviews.groupby("order_id", as_index=False).agg(
        review_score_mean=("review_score", "mean"),
        review_count=("review_id", "count"),
    )

    orders = orders.merge(customers, on="customer_id", validate="many_to_one")
    orders = orders.merge(payment_totals, on="order_id", how="left", validate="one_to_one")
    orders = orders.merge(review_totals, on="order_id", how="left", validate="one_to_one")
    orders["purchase_date"] = orders["order_purchase_timestamp"].dt.strftime("%Y-%m-%d")

    products = products.merge(categories, on="product_category_name", how="left", validate="many_to_one")
    products["product_category_name_english"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("unknown")
    )

    order_items = items.merge(orders, on="order_id", validate="many_to_one")
    order_items = order_items.merge(products, on="product_id", validate="many_to_one")
    order_items = order_items.merge(sellers, on="seller_id", validate="many_to_one")

    interactions = order_items.loc[
        order_items["order_status"].eq("delivered"),
        [
            "order_id",
            "order_item_id",
            "customer_unique_id",
            "product_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "purchase_date",
        ],
    ].copy()
    interactions["interaction_type"] = "purchase"

    daily_demand = (
        order_items.groupby(
            ["purchase_date", "product_id", "customer_state"], as_index=False
        )
        .size()
        .rename(columns={"size": "order_item_count"})
    )

    return {
        "orders": orders,
        "order_items": order_items,
        "purchase_interactions": interactions,
        "daily_demand": daily_demand,
    }


def select_partition(
    tables: dict[str, pd.DataFrame], purchase_date: str | None
) -> dict[str, pd.DataFrame]:
    if not purchase_date:
        return tables
    date.fromisoformat(purchase_date)
    selected = {
        name: frame.loc[frame["purchase_date"].eq(purchase_date)].copy()
        for name, frame in tables.items()
    }
    if selected["orders"].empty:
        raise ValueError(f"No orders found for purchase date {purchase_date}")
    return selected


def write_partition_files(
    tables: dict[str, pd.DataFrame], output_dir: Path, purchase_date: str | None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = [purchase_date] if purchase_date else tables["orders"]["purchase_date"].dropna().unique()
    for partition in dates:
        partition_dir = output_dir / f"purchase_date={partition}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.loc[frame["purchase_date"].eq(partition)].to_csv(
                partition_dir / f"{name}.csv", index=False
            )


def write_warehouse(
    tables: dict[str, pd.DataFrame], warehouse: Path, purchase_date: str | None
) -> None:
    warehouse.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(warehouse)) as connection:
        for name, frame in tables.items():
            warehouse_frame = frame.astype(object).where(pd.notna(frame), None)
            connection.register("curated_frame", warehouse_frame)
            exists = connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [name],
            ).fetchone()[0]
            if purchase_date and exists:
                connection.execute(f'DELETE FROM "{name}" WHERE purchase_date = ?', [purchase_date])
                connection.execute(f'INSERT INTO "{name}" SELECT * FROM curated_frame')
            else:
                connection.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM curated_frame')
            connection.unregister("curated_frame")


def write_outputs(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    warehouse: Path,
    purchase_date: str | None,
) -> dict[str, int]:
    write_partition_files(tables, output_dir, purchase_date)
    write_warehouse(tables, warehouse, purchase_date)

    return {name: len(frame) for name, frame in tables.items()}


def quality_report(
    tables: dict[str, pd.DataFrame], row_counts: dict[str, int], purchase_date: str | None
) -> dict:
    checks = {
        "orders_have_customers": int(tables["orders"]["customer_id"].notna().sum()),
        "items_have_orders": int(tables["order_items"]["order_id"].notna().sum()),
        "items_have_products": int(tables["order_items"]["product_id"].notna().sum()),
        "items_have_sellers": int(tables["order_items"]["seller_id"].notna().sum()),
    }
    return {
        "status": "success",
        "purchase_date": purchase_date,
        "tables": row_counts,
        "checks": checks,
        "warehouse_tables": list(tables),
    }


def main() -> int:
    args = parse_args()
    tables = select_partition(build_tables(args.input_dir), args.purchase_date)
    row_counts = write_outputs(tables, args.output_dir, args.warehouse, args.purchase_date)
    report = quality_report(tables, row_counts, args.purchase_date)

    report_path = PROCESSED / "quality_report.json"
    status_path = PROCESSED / "pipeline_status.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    status_path.write_text(
        json.dumps({"status": report["status"], "stage": "curated_data"}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
