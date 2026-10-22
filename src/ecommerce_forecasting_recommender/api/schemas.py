from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    day_index: int = Field(ge=0)
    lag_1: float = Field(ge=0)
    avg_price: float = Field(ge=0)


class ForecastResponse(BaseModel):
    prediction: float


class RecommendationRequest(BaseModel):
    customer_id: str
    top_k: int = Field(default=5, ge=1, le=50)


class RecommendationResponse(BaseModel):
    recommendations: list[str]
