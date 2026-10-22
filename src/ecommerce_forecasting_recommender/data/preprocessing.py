from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_ORDERS = "olist_orders_dataset.csv"
RAW_ORDER_ITEMS = "olist_order_items_dataset.csv"
RAW_PRODUCTS = "olist_products_dataset.csv"


def build_modeling_table(raw_dir: Path) -> pd.DataFrame:
    orders = pd.read_csv(raw_dir / RAW_ORDERS)
    items = pd.read_csv(raw_dir / RAW_ORDER_ITEMS)
    products = pd.read_csv(raw_dir / RAW_PRODUCTS)

    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

    merged = (
        orders.merge(items[["order_id", "product_id", "price"]], on="order_id", how="inner")
        .merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
        .sort_values("order_purchase_timestamp")
    )
    merged["order_date"] = merged["order_purchase_timestamp"].dt.date.astype("string")
    merged["quantity"] = 1
    merged["product_category_name"] = merged["product_category_name"].fillna("unknown")
    return merged


def save_processed_dataset(data: pd.DataFrame, processed_path: Path) -> Path:
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed_path, index=False)
    return processed_path
