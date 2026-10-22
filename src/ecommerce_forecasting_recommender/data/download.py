from __future__ import annotations

import logging
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from ecommerce_forecasting_recommender.utils.io import ensure_directory

LOGGER = logging.getLogger(__name__)

EXPECTED_FILES = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_products_dataset.csv",
]


def bootstrap_sample_raw_data(raw_dir: Path) -> list[Path]:
    """Create lightweight sample files when real download is unavailable."""
    ensure_directory(raw_dir)

    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "customer_id": ["c1", "c1", "c2", "c3"],
            "order_purchase_timestamp": [
                "2018-01-01 10:00:00",
                "2018-01-03 11:00:00",
                "2018-01-04 15:00:00",
                "2018-01-05 09:30:00",
            ],
        }
    )
    order_items = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "product_id": ["p1", "p2", "p1", "p3"],
            "order_item_id": [1, 1, 1, 1],
            "price": [10.0, 30.0, 10.0, 22.0],
        }
    )
    customers = pd.DataFrame({"customer_id": ["c1", "c2", "c3"], "customer_unique_id": ["u1", "u2", "u3"]})
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "product_category_name": ["books", "furniture", "toys"],
        }
    )

    files = {
        "olist_orders_dataset.csv": orders,
        "olist_order_items_dataset.csv": order_items,
        "olist_customers_dataset.csv": customers,
        "olist_products_dataset.csv": products,
    }
    written: list[Path] = []
    for name, frame in files.items():
        path = raw_dir / name
        frame.to_csv(path, index=False)
        written.append(path)
    LOGGER.info("Sample Olist-compatible data generated in %s", raw_dir)
    return written


def extract_downloaded_zip(zip_path: Path, raw_dir: Path) -> list[Path]:
    """Extract official Olist zip data previously downloaded by the user."""
    ensure_directory(raw_dir)
    written: list[Path] = []
    with ZipFile(zip_path, "r") as archive:
        for member in archive.namelist():
            if member.endswith(".csv"):
                archive.extract(member, path=raw_dir)
                written.append(raw_dir / member)
    LOGGER.info("Extracted %d CSV files from %s", len(written), zip_path)
    return written
