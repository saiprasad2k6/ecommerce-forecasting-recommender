from pathlib import Path

from ecommerce_forecasting_recommender.data.download import bootstrap_sample_raw_data
from ecommerce_forecasting_recommender.pipelines.train import run_training_pipeline


def test_training_pipeline_writes_artifacts(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    bootstrap_sample_raw_data(raw)

    metrics = run_training_pipeline(raw_dir=raw, processed_dir=processed, model_dir=models)

    assert "forecast_model_mae" in metrics
    assert (models / "forecast_model.pkl").exists()
    assert (models / "recommender.pkl").exists()
    assert (processed / "demand_features.csv").exists()
