from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ecommerce-forecasting-recommender"
    model_dir: Path = Path("models")
    processed_data_dir: Path = Path("data/processed")
    raw_data_dir: Path = Path("data/raw")
    reference_data_dir: Path = Path("data/reference")
    mlflow_tracking_uri: str = "file:./mlruns"
    top_k_recommendations: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
