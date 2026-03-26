"""
API Schemas — full Pydantic v2 models with OpenAPI examples.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    CRITICAL = "CRITICAL"

class ModelStage(str, Enum):
    STAGING    = "Staging"
    PRODUCTION = "Production"
    ARCHIVED   = "Archived"


# ─── Transaction / Prediction ─────────────────────────────────────────────────

class TransactionFeatures(BaseModel):
    """Raw input features for a single transaction."""

    V1:  float = Field(..., description="PCA component 1")
    V2:  float = Field(..., description="PCA component 2")
    V3:  float = Field(..., description="PCA component 3")
    V4:  float = Field(..., description="PCA component 4")
    V5:  float = Field(..., description="PCA component 5")
    V6:  float = Field(..., description="PCA component 6")
    V7:  float = Field(..., description="PCA component 7")
    V8:  float = Field(..., description="PCA component 8")
    V9:  float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., ge=0.0, description="Transaction amount (NGN / USD)")
    Time:   float = Field(..., ge=0.0, description="Seconds since first transaction in dataset")

    model_config = {
        "json_schema_extra": {
            "example": {
                "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
                "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
                "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
                "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
                "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
                "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
                "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
                "Amount": 149.62, "Time": 0.0
            }
        }
    }


class PredictRequest(BaseModel):
    transaction_id: Optional[str] = Field(None, description="Optional client-side idempotency key")
    transaction:    TransactionFeatures

    model_config = {"json_schema_extra": {"example": {
        "transaction_id": "TXN-2024-000123",
        "transaction": TransactionFeatures.model_config["json_schema_extra"]["example"]
    }}}


class PredictResponse(BaseModel):
    transaction_id:   str
    is_fraud:         bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level:       RiskLevel
    confidence:       str
    threshold_used:   float
    model_name:       str
    model_version:    str
    latency_ms:       float
    timestamp:        datetime


class BatchPredictRequest(BaseModel):
    transactions: List[TransactionFeatures] = Field(..., min_length=1, max_length=10_000)

    @field_validator("transactions")
    @classmethod
    def check_limit(cls, v):
        if len(v) > 10_000:
            raise ValueError("Batch limit is 10,000 transactions")
        return v


class BatchPredictResponse(BaseModel):
    total:           int
    fraud_count:     int
    fraud_rate_pct:  float
    results:         List[PredictResponse]
    model_name:      str
    latency_ms:      float
    timestamp:       datetime


# ─── Model registry ──────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name:          str
    version:       str
    stage:         str
    run_id:        str
    metrics:       Dict[str, float]
    params:        Dict[str, Any]
    tags:          Dict[str, str]
    created_at:    Optional[datetime]
    description:   Optional[str]


class ModelPromoteRequest(BaseModel):
    model_name:    str
    version:       str
    target_stage:  ModelStage
    justification: str = Field(..., min_length=10, description="Reason for promotion")

    model_config = {"json_schema_extra": {"example": {
        "model_name": "fraud-detector-champion",
        "version": "7",
        "target_stage": "Production",
        "justification": "AUPRC improved from 0.873 to 0.901 on holdout set. Approved by ML team."
    }}}


class ModelRegisterRequest(BaseModel):
    run_id:        str = Field(..., description="MLflow run ID")
    model_name:    str = Field(..., description="Registry model name")
    description:   Optional[str] = None
    tags:          Optional[Dict[str, str]] = {}


# ─── Experiments ─────────────────────────────────────────────────────────────

class ExperimentRun(BaseModel):
    run_id:        str
    run_name:      Optional[str]
    status:        str
    metrics:       Dict[str, float]
    params:        Dict[str, Any]
    tags:          Dict[str, str]
    start_time:    Optional[datetime]
    end_time:      Optional[datetime]
    artifact_uri:  str


class ExperimentSummary(BaseModel):
    experiment_id:   str
    name:            str
    lifecycle_stage: str
    runs:            List[ExperimentRun]


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthStatus(str, Enum):
    OK       = "ok"
    DEGRADED = "degraded"
    DOWN     = "down"


class DependencyHealth(BaseModel):
    name:    str
    status:  HealthStatus
    latency_ms: Optional[float]
    detail:  Optional[str]


class HealthResponse(BaseModel):
    status:       HealthStatus
    version:      str
    environment:  str
    model:        Dict[str, Any]
    dependencies: List[DependencyHealth]
    timestamp:    datetime


# ─── Auth ─────────────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    username: str
    password: str

    model_config = {"json_schema_extra": {"example": {"username": "admin", "password": "admin"}}}


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int


# ─── Transactions ─────────────────────────────────────────────────────────────

class TransactionRecord(BaseModel):
    id:               int
    transaction_id:   str
    is_fraud:         bool
    fraud_probability: float
    risk_level:       str
    model_name:       str
    model_version:    str
    latency_ms:       float
    created_at:       datetime

    model_config = {"from_attributes": True}


class PaginatedTransactions(BaseModel):
    total:  int
    page:   int
    size:   int
    items:  List[TransactionRecord]


# ─── Drift ───────────────────────────────────────────────────────────────────

class DriftReport(BaseModel):
    timestamp:        datetime
    features_drifted: List[str]
    drift_scores:     Dict[str, float]
    alert_triggered:  bool
    recommendation:   str
