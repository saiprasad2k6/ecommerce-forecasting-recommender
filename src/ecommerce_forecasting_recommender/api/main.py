from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import generate_latest

from ecommerce_forecasting_recommender.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from ecommerce_forecasting_recommender.config import settings
from ecommerce_forecasting_recommender.models.forecasting import load_forecast_model, predict_demand
from ecommerce_forecasting_recommender.models.recommender import (
    load_recommender,
    recommend_products,
)
from ecommerce_forecasting_recommender.monitoring.metrics import latency

app = FastAPI(title="ecommerce-forecasting-recommender")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(payload: ForecastRequest) -> ForecastResponse:
    with latency("forecast"):
        model_path = settings.model_dir / "forecast_model.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=503, detail="Forecast model is not available")
        model = load_forecast_model(model_path)
        prediction = predict_demand(model, payload.day_index, payload.lag_1, payload.avg_price)
        return ForecastResponse(prediction=prediction)


@app.post("/recommendations", response_model=RecommendationResponse)
def recommendations(payload: RecommendationRequest) -> RecommendationResponse:
    with latency("recommendations"):
        model_path = settings.model_dir / "recommender.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=503, detail="Recommender model is not available")
        model = load_recommender(model_path)
        return RecommendationResponse(
            recommendations=recommend_products(model, payload.customer_id, top_k=payload.top_k)
        )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")
