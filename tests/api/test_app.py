from pathlib import Path

from fastapi.testclient import TestClient

from ecommerce_forecasting_recommender.api.main import app
from ecommerce_forecasting_recommender.config import settings
from ecommerce_forecasting_recommender.data.download import bootstrap_sample_raw_data
from ecommerce_forecasting_recommender.pipelines.train import run_training_pipeline


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_forecast_and_recommendations(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    model_dir = tmp_path / "models"
    bootstrap_sample_raw_data(raw)
    run_training_pipeline(raw_dir=raw, processed_dir=processed, model_dir=model_dir)

    monkeypatch.setattr(settings, "model_dir", model_dir)

    client = TestClient(app)
    forecast_response = client.post("/forecast", json={"day_index": 5, "lag_1": 1.0, "avg_price": 20.0})
    assert forecast_response.status_code == 200
    assert forecast_response.json()["prediction"] >= 0

    rec_response = client.post("/recommendations", json={"customer_id": "c1", "top_k": 3})
    assert rec_response.status_code == 200
    assert isinstance(rec_response.json()["recommendations"], list)
