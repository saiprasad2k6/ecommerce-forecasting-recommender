from pathlib import Path

from ecommerce_forecasting_recommender.data.download import bootstrap_sample_raw_data
from ecommerce_forecasting_recommender.data.preprocessing import build_modeling_table


def test_build_modeling_table_contains_expected_columns(tmp_path: Path) -> None:
    bootstrap_sample_raw_data(tmp_path)
    frame = build_modeling_table(tmp_path)
    assert {"order_id", "customer_id", "product_id", "quantity", "order_purchase_timestamp"}.issubset(frame.columns)
    assert len(frame) > 0
