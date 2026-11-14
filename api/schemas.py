"""Validated request and response models used to generate the OpenAPI contract."""
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^\S+$")]
StateCode = Literal[
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG",
    "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
]


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastItem(RequestModel):
    product_id: Identifier
    customer_state: StateCode


class ForecastRequest(RequestModel):
    as_of: date = Field(description="Prepared feature date; predicts demand for the following day.")
    items: list[ForecastItem] = Field(min_length=1, max_length=100)
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "as_of": "2018-08-01", "items": [
            {"product_id": "0c44cf18fca971245b41119ce72861c0", "customer_state": "SP"}
        ],
    }})

    @model_validator(mode="after")
    def unique_items(self):
        keys = [(item.product_id, item.customer_state) for item in self.items]
        if len(set(keys)) != len(keys):
            raise ValueError("Each product/state pair must be unique")
        return self


class RecommendationRequest(RequestModel):
    as_of: date = Field(description="Use delivered purchase history strictly before this date.")
    customer_unique_id: Identifier
    k: int | None = Field(default=None, ge=1, le=100, strict=True,
                          description="Number of recommendations; defaults to the active model's k. Must not exceed its candidate limit.")
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "as_of": "2018-08-01", "customer_unique_id": "0000366f3b9a7992bf8c76cfdf3221e2", "k": 10,
    }})


class ModelInfo(BaseModel):
    task: Literal["forecasting", "recommendation"]
    model_name: str
    model_role: Literal["baseline", "candidate"]
    model_run_id: str
    training_pipeline_run_id: str


class ForecastPrediction(ForecastItem):
    target_date: date
    prediction: float = Field(ge=0, allow_inf_nan=False, description="Estimated demand, not observed sales.")


class ForecastResponse(BaseModel):
    request_id: str
    generated_at: datetime
    as_of: date
    model: ModelInfo
    predictions: list[ForecastPrediction]


class RecommendedProduct(BaseModel):
    rank: int = Field(ge=1)
    product_id: str


class RecommendationResponse(BaseModel):
    request_id: str
    generated_at: datetime
    as_of: date
    customer_unique_id: str
    k: int
    cold_start: bool
    model: ModelInfo
    recommendations: list[RecommendedProduct]


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[str] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
