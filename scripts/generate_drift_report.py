from pathlib import Path

from ecommerce_forecasting_recommender.monitoring.drift import generate_simple_drift_report

if __name__ == "__main__":
    output = generate_simple_drift_report(
        reference_path=Path("data/reference/reference_demand.csv"),
        current_path=Path("data/processed/demand_features.csv"),
        output_path=Path("artifacts/drift_report.json"),
    )
    print(f"Saved drift report to {output}")
