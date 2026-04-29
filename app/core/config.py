from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API
    API_VERSION: str = "2.1.0"
    ENVIRONMENT: str = "development"
    SECRET_KEY: str = "change-me-in-production-32-chars!!"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # MongoDB 
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "frauddb"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "fraud-detection"
    CHAMPION_MODEL_NAME: str = "fraud-detector-champion"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    PREDICTION_CACHE_TTL: int = 300

    # Model
    MODEL_PATH: str = "./model"
    PREDICTION_THRESHOLD: float = 0.33
    BATCH_LIMIT: int = 10_000

    # Monitoring
    DRIFT_ALERT_THRESHOLD: float = 0.15
    LOG_LEVEL: str = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
