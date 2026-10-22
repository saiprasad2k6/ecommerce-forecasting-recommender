from pathlib import Path

from ecommerce_forecasting_recommender.data.validation import validate_raw_data

if __name__ == "__main__":
    report = validate_raw_data(Path("data/raw"))
    if report["missing_files"] or report["invalid_columns"]:
        raise SystemExit(f"Validation failed: {report}")
    print("Validation passed")
