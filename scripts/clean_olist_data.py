#!/usr/bin/env python3
"""Clean Olist CSV files without modifying data/raw."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

TABLES = {
    "customers": (
        "olist_customers_dataset.csv",
        ["customer_id"],
        ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"],
    ),
    "orders": (
        "olist_orders_dataset.csv",
        ["order_id"],
        ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
    ),
    "order_items": (
        "olist_order_items_dataset.csv",
        ["order_id", "order_item_id"],
        ["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price", "freight_value"],
    ),
    "products": ("olist_products_dataset.csv", ["product_id"], ["product_id"]),
    "sellers": ("olist_sellers_dataset.csv", ["seller_id"], ["seller_id"]),
    "payments": (
        "olist_order_payments_dataset.csv",
        ["order_id", "payment_sequential"],
        ["order_id", "payment_sequential"],
    ),
    "reviews": (
        "olist_order_reviews_dataset.csv",
        ["review_id", "order_id"],
        ["review_id", "order_id"],
    ),
    "categories": (
        "product_category_name_translation.csv",
        ["product_category_name"],
        ["product_category_name"],
    ),
    "geolocation": (
        "olist_geolocation_dataset.csv",
        [],
        ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng", "geolocation_city", "geolocation_state"],
    ),
}

NUMERIC = {
    "order_item_id", "price", "freight_value", "payment_sequential",
    "payment_installments", "payment_value", "review_score",
    "product_name_lenght", "product_description_lenght", "product_photos_qty",
    "product_weight_g", "product_length_cm", "product_height_cm",
    "product_width_cm", "geolocation_lat", "geolocation_lng",
}
TIMESTAMPS = {
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
    "shipping_limit_date", "review_creation_date", "review_answer_timestamp",
}
INTEGER_COLUMNS = NUMERIC - {"price", "freight_value", "payment_value", "geolocation_lat", "geolocation_lng"}
COORDINATES = {"geolocation_lat", "geolocation_lng"}
VALID_STATUSES = {"delivered", "invoiced", "shipped", "processing", "unavailable", "canceled", "created", "approved"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    return parser.parse_args()


def add_reason(reasons: pd.Series, mask: pd.Series, reason: str) -> None:
    reasons.loc[mask.fillna(False)] += reason + ";"


def quarantine(rows: pd.DataFrame, reason: str, filename: str) -> pd.DataFrame:
    result = rows.copy()
    result["source_file"] = filename
    result["failure_reason"] = reason
    return result


def schema_for(frame: pd.DataFrame, required_columns: list[str]) -> pa.DataFrameSchema:
    columns = {}
    for column in frame.columns:
        if column == "_source_row_number":
            continue
        nullable = column not in required_columns
        if column in NUMERIC:
            checks = []
            if column not in COORDINATES:
                checks.append(pa.Check.ge(0))
            if column in INTEGER_COLUMNS:
                checks.append(pa.Check(lambda values: (values % 1) == 0))
            if column == "review_score":
                checks.append(pa.Check.le(5))
            dtype = pa.Int64 if column in INTEGER_COLUMNS else pa.Float64
            columns[column] = pa.Column(dtype, checks=checks, nullable=nullable)
        elif column in TIMESTAMPS:
            columns[column] = pa.Column(pa.DateTime, nullable=nullable)
        else:
            columns[column] = pa.Column(pa.String, nullable=nullable)
    if "order_status" in frame:
        columns["order_status"] = pa.Column(
            pa.String,
            checks=pa.Check.isin(VALID_STATUSES),
            nullable=False,
        )
    return pa.DataFrameSchema(columns, strict=False)


def clean_table(name: str, input_dir: Path, quarantine_rows: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, int]]:
    filename, keys, required_columns = TABLES[name]
    frame = pd.read_csv(input_dir / filename, dtype="string", encoding="utf-8-sig")
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

    frame["_source_row_number"] = frame.index + 2
    for column in frame.columns:
        if column != "_source_row_number":
            frame[column] = frame[column].str.strip().replace("", pd.NA)

    reasons = pd.Series("", index=frame.index, dtype="string")
    for column in frame.columns:
        if column.startswith("_"):
            continue
        if column in required_columns:
            add_reason(reasons, frame[column].isna(), f"{column}:required")
        if column in NUMERIC:
            parsed = pd.to_numeric(frame[column], errors="coerce")
            add_reason(reasons, frame[column].notna() & parsed.isna(), f"{column}:numeric")
            if column not in COORDINATES:
                add_reason(reasons, parsed < 0, f"{column}:minimum")
            if column in INTEGER_COLUMNS:
                add_reason(reasons, parsed.notna() & (parsed % 1 != 0), f"{column}:integer")
            frame[column] = parsed
        elif column in TIMESTAMPS:
            parsed = pd.to_datetime(frame[column], format="%Y-%m-%d %H:%M:%S", errors="coerce")
            add_reason(reasons, frame[column].notna() & parsed.isna(), f"{column}:timestamp")
            frame[column] = parsed
        elif column.endswith("_city") or column.startswith("product_category_name"):
            frame[column] = frame[column].str.lower()
        elif column.endswith("_zip_code_prefix"):
            frame[column] = frame[column].str.zfill(5)

    if name == "orders":
        add_reason(reasons, ~frame.order_status.isin(VALID_STATUSES), "order_status:unknown")

    invalid = reasons.ne("")
    if invalid.any():
        quarantine_rows.append(quarantine(frame.loc[invalid], reasons.loc[invalid].str.rstrip(";"), filename))
    frame = frame.loc[~invalid].copy()

    try:
        schema_for(frame, required_columns).validate(frame, lazy=True)
    except pa.errors.SchemaErrors as error:
        raise ValueError(f"{filename} failed Pandera validation:\n{error.failure_cases}") from error

    for column in TIMESTAMPS & set(frame.columns):
        frame[column] = frame[column].dt.strftime("%Y-%m-%d %H:%M:%S")

    data_columns = frame.columns.drop("_source_row_number")
    exact_duplicates = frame.duplicated(subset=data_columns, keep="first")
    if exact_duplicates.any():
        quarantine_rows.append(quarantine(frame.loc[exact_duplicates], "exact_duplicate", filename))
        frame = frame.loc[~exact_duplicates].copy()

    key_duplicates = pd.Series(False, index=frame.index)
    if keys:
        key_duplicates = frame.duplicated(keys, keep="first")
        if key_duplicates.any():
            quarantine_rows.append(quarantine(frame.loc[key_duplicates], "duplicate_primary_key", filename))
            frame = frame.loc[~key_duplicates].copy()

    metrics = {
        "input_rows": len(reasons),
        "invalid_rows": int(invalid.sum()),
        "exact_duplicates_removed": int(exact_duplicates.sum()),
        "key_duplicates_removed": int(key_duplicates.sum()),
        "output_rows": len(frame),
    }
    return frame, metrics


def remove_orphans(
    tables: dict[str, pd.DataFrame],
    quarantine_rows: list[pd.DataFrame],
    metrics: dict[str, dict[str, int]],
) -> None:
    relations = [
        ("orders", "customer_id", "customers"),
        ("order_items", "order_id", "orders"),
        ("order_items", "product_id", "products"),
        ("order_items", "seller_id", "sellers"),
        ("payments", "order_id", "orders"),
        ("reviews", "order_id", "orders"),
    ]
    for child, column, parent in relations:
        invalid = ~tables[child][column].isin(tables[parent][column])
        if invalid.any():
            quarantine_rows.append(
                quarantine(
                    tables[child].loc[invalid],
                    f"{column}:missing_{parent}",
                    TABLES[child][0],
                )
            )
            tables[child] = tables[child].loc[~invalid].copy()
            metrics[child]["orphan_rows_removed"] = metrics[child].get("orphan_rows_removed", 0) + int(invalid.sum())


def main() -> int:
    args = parse_args()
    tables: dict[str, pd.DataFrame] = {}
    quarantine_rows: list[pd.DataFrame] = []
    metrics: dict[str, dict[str, int]] = {}

    for name in TABLES:
        tables[name], metrics[name] = clean_table(name, args.input_dir, quarantine_rows)
    remove_orphans(tables, quarantine_rows, metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir = args.output_dir / "quarantine"
    quarantine_dir.mkdir(exist_ok=True)
    for name, frame in tables.items():
        output = frame.drop(columns="_source_row_number")
        output.to_csv(args.output_dir / f"{name}.csv", index=False)
    if quarantine_rows:
        rejected = pd.concat(quarantine_rows, ignore_index=True)
        rejected.to_csv(quarantine_dir / "rejected_records.csv", index=False)

    with (args.output_dir / "clean_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)

    print(json.dumps({
        "status": "success",
        "processed_dir": str(args.output_dir),
        "quarantined_rows": sum(len(rows) for rows in quarantine_rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
