from pathlib import Path

from ecommerce_forecasting_recommender.data.download import bootstrap_sample_raw_data
from ecommerce_forecasting_recommender.data.validation import validate_raw_data


def test_validate_raw_data_passes_with_sample_data(tmp_path: Path) -> None:
    bootstrap_sample_raw_data(tmp_path)
    report = validate_raw_data(tmp_path)
    assert report["missing_files"] == []
    assert report["invalid_columns"] == []
