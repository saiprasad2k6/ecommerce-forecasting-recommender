"""FastAPI inference server. Run: uvicorn api.app:app --host 127.0.0.1 --port 8000."""
from datetime import datetime, timezone
import logging
from uuid import uuid4

import duckdb
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import (
    ErrorResponse, ForecastRequest, ForecastResponse, HealthResponse,
    ModelsResponse, RecommendationRequest, RecommendationResponse,
)
from api.service import InferenceError, InferenceService

LOGGER = logging.getLogger(__name__)
ERRORS = {
    404: {"model": ErrorResponse, "description": "Requested prepared features or historical data were not found."},
    422: {"model": ErrorResponse, "description": "Invalid request fields or recommendation k."},
    503: {"model": ErrorResponse, "description": "Model/input unavailable, invalid, locked, or workspace busy."},
}


def create_app(service=None):
    service = service or InferenceService()
    application = FastAPI(
        title="Olist Inference API", version="1.0.0",
        description=("Request-time inference using activated model bundles. Forecasts read prepared "
                     "features for as_of and predict the following day. Recommendations use delivered "
                     "history before as_of. Responses identify the actual serving model. "
                     "Requests do not train models or write batch predictions to DuckDB."),
        openapi_tags=[{"name": "Operations"}, {"name": "Inference"}],
    )

    @application.exception_handler(InferenceError)
    async def inference_error(request: Request, error: InferenceError):
        body = ErrorResponse(error={"code": error.code, "message": str(error), "details": error.details})
        return JSONResponse(status_code=error.status, content=body.model_dump())

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        details = [f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in error.errors()]
        body = ErrorResponse(error={"code": "validation_error", "message": "Invalid request.", "details": details})
        return JSONResponse(status_code=422, content=body.model_dump())

    def execute(operation):
        try:
            return operation()
        except InferenceError:
            raise
        except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError, duckdb.Error) as error:
            LOGGER.warning("Inference resources unavailable: %s", error)
            raise InferenceError(503, "inference_unavailable",
                                 "Inference resources are unavailable. Check active models, input files and workspace/database locks.") from error

    @application.get("/health", response_model=HealthResponse, tags=["Operations"], operation_id="health",
                     summary="Check that the HTTP service is running")
    def health():
        return {"status": "ok"}

    @application.get("/v1/models", response_model=ModelsResponse, tags=["Operations"],
                     responses={503: ERRORS[503]}, operation_id="active_models",
                     summary="Inspect the activated models and check serving readiness")
    def active_models():
        return {"models": execute(service.models)}

    @application.post("/v1/inference/forecast", response_model=ForecastResponse, tags=["Inference"],
                      responses=ERRORS, operation_id="forecast_demand", summary="Forecast next-day product/state demand")
    def forecast(body: ForecastRequest):
        result = execute(lambda: service.forecast(body))
        return {"request_id": uuid4().hex, "generated_at": datetime.now(timezone.utc), "as_of": body.as_of, **result}

    @application.post("/v1/inference/recommendations", response_model=RecommendationResponse, tags=["Inference"],
                      responses=ERRORS, operation_id="recommend_products", summary="Recommend ranked products for one customer",
                      description="Unknown customers use the active model's cold-start behavior. Fewer than k products may be returned if the catalog is smaller. No scores are exposed.")
    def recommendations(body: RecommendationRequest):
        result = execute(lambda: service.recommend(body))
        return {"request_id": uuid4().hex, "generated_at": datetime.now(timezone.utc),
                "as_of": body.as_of, "customer_unique_id": body.customer_unique_id, **result}

    return application


app = create_app()
