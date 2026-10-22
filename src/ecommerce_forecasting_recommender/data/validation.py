from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "olist_orders_dataset.csv": {"order_id", "customer_id", "order_purchase_timestamp"},
    "olist_order_items_dataset.csv": {"order_id", "product_id", "order_item_id"},
    "olist_customers_dataset.csv": {"customer_id", "customer_unique_id"},
    "olist_products_dataset.csv": {"product_id", "product_category_name"},
}


def validate_raw_data(raw_dir: Path) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {"missing_files": [], "invalid_columns": []}
    for file_name, expected_cols in REQUIRED_COLUMNS.items():
        target = raw_dir / file_name
        if not target.exists():
            report["missing_files"].append(file_name)
            continue
        frame = pd.read_csv(target, nrows=100)
        missing_cols = sorted(expected_cols.difference(frame.columns))
        if missing_cols:
            report["invalid_columns"].append(f"{file_name}: {', '.join(missing_cols)}")
    return report
