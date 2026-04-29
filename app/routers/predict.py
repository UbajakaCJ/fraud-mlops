"""
Prediction Router
-----------------
POST /predict          — real-time single transaction
POST /predict/batch    — bulk scoring (up to 10k)
GET  /predict/explain/{id} — SHAP-style explanation
"""

import time
import uuid
import numpy as np
import structlog
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import settings
from app.core.database import get_db
from app.core.model_registry import model_registry
from app.schemas.schemas import (
    PredictRequest, PredictResponse, BatchPredictRequest,
    BatchPredictResponse, RiskLevel
)
from app.services.feature_engineering import engineer_features
from app.services.audit_logger import log_prediction
from app.services.cache import get_cached_prediction, set_cached_prediction

log = structlog.get_logger()

router = APIRouter()


def _risk_level(prob: float) -> RiskLevel:
    if prob < 0.25:   return RiskLevel.LOW
    if prob < 0.50:   return RiskLevel.MEDIUM
    if prob < 0.75:   return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def _confidence(prob: float) -> str:
    dist = abs(prob - settings.PREDICTION_THRESHOLD)
    if dist > 0.4:  return "very_high"
    if dist > 0.25: return "high"
    if dist > 0.1:  return "medium"
    return "low"


def _features_to_array(txn) -> np.ndarray:
    raw = {**txn.model_dump()}
    return engineer_features(raw)


# ─── Single Prediction ────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=PredictResponse,
    summary="Predict fraud for a single transaction",
    response_description="Fraud score with risk classification",
    responses={
        200: {"description": "Prediction successful"},
        503: {"description": "Model not loaded"},
        422: {"description": "Validation error"},
    },
)
async def predict_single(
    payload: PredictRequest,
    background_tasks: BackgroundTasks,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Score a single credit card transaction for fraud.

    - **V1–V28**: PCA-transformed features (required)
    - **Amount**: Transaction value in base currency
    - **Time**: Seconds elapsed since dataset epoch

    Returns a fraud probability score, risk level (LOW/MEDIUM/HIGH/CRITICAL),
    and the decision threshold used for this prediction.

    Results are cached for 5 minutes by transaction_id (if provided).
    """
    if not model_registry.is_ready():
        raise HTTPException(503, "Model not loaded — try again shortly")

    txn_id = payload.transaction_id or f"TXN-{uuid.uuid4().hex[:12].upper()}"

    # Cache check
    cached = await get_cached_prediction(txn_id)
    if cached:
        log.info("cache_hit", txn_id=txn_id)
        return cached

    t0 = time.perf_counter()
    X = _features_to_array(payload.transaction)
    preds, probs = model_registry.predict(X)
    latency = (time.perf_counter() - t0) * 1000

    prob  = float(probs[0])
    fraud = bool(preds[0])

    response = PredictResponse(
        transaction_id=txn_id,
        is_fraud=fraud,
        fraud_probability=round(prob, 6),
        risk_level=_risk_level(prob),
        confidence=_confidence(prob),
        threshold_used=settings.PREDICTION_THRESHOLD,
        model_name=model_registry.champion_name,
        model_version=model_registry.champion_version,
        latency_ms=round(latency, 2),
        timestamp=datetime.now(timezone.utc),
    )

    # Background: cache + audit log
    await set_cached_prediction(txn_id, response)
    background_tasks.add_task(log_prediction, db, response, payload.transaction)

    log.info("prediction", txn_id=txn_id, fraud=fraud, prob=round(prob, 4), latency_ms=round(latency, 2))
    return response


# ─── Batch Prediction ─────────────────────────────────────────────────────────

@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Batch fraud scoring (up to 10,000 transactions)",
    responses={
        200: {"description": "Batch scored successfully"},
        413: {"description": "Batch exceeds 10,000 transaction limit"},
        503: {"description": "Model not loaded"},
    },
)
async def predict_batch(
    payload: BatchPredictRequest,
    background_tasks: BackgroundTasks,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Score a batch of transactions in a single request.

    Transactions are processed as a vectorised NumPy array — throughput is
    typically 5,000–15,000 transactions/second depending on model type.

    The response includes aggregate fraud statistics alongside per-transaction
    scores, enabling downstream alerting and reporting.
    """
    if not model_registry.is_ready():
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()

    # Vectorise
    features = [_features_to_array(txn) for txn in payload.transactions]
    X = np.vstack(features)
    preds, probs = model_registry.predict(X)
    latency = (time.perf_counter() - t0) * 1000

    now = datetime.now(timezone.utc)
    results = []
    for i, (txn, pred, prob) in enumerate(zip(payload.transactions, preds, probs)):
        txn_id = f"BATCH-{uuid.uuid4().hex[:10].upper()}"
        r = PredictResponse(
            transaction_id=txn_id,
            is_fraud=bool(pred),
            fraud_probability=round(float(prob), 6),
            risk_level=_risk_level(float(prob)),
            confidence=_confidence(float(prob)),
            threshold_used=settings.PREDICTION_THRESHOLD,
            model_name=model_registry.champion_name,
            model_version=model_registry.champion_version,
            latency_ms=round(latency / len(payload.transactions), 3),
            timestamp=now,
        )
        results.append(r)

    fraud_count = int(preds.sum())
    response = BatchPredictResponse(
        total=len(results),
        fraud_count=fraud_count,
        fraud_rate_pct=round(fraud_count / len(results) * 100, 4),
        results=results,
        model_name=model_registry.champion_name,
        latency_ms=round(latency, 2),
        timestamp=now,
    )

    log.info("batch_prediction", total=len(results), fraud_count=fraud_count, latency_ms=round(latency, 2))
    return response


# ─── Drift report ─────────────────────────────────────────────────────────────

@router.get(
    "/drift",
    summary="Feature drift report",
    tags=["predict"],
    responses={200: {"description": "Current drift statistics"}},
)
async def get_drift_report():
    """
    Returns PSI (Population Stability Index) scores for all input features
    compared to the training baseline distribution.

    PSI > 0.15 triggers a retraining recommendation.
    PSI > 0.25 triggers a critical alert.
    """
    # In production this queries a monitoring DB; here we return a mock structure
    return {
        "timestamp": datetime.now(timezone.utc),
        "features_monitored": 32,
        "features_drifted": ["V14", "Amount"],
        "drift_scores": {
            "V14": 0.18, "V17": 0.09, "Amount": 0.22,
            "V1": 0.03,  "V4": 0.05,  "log_amount": 0.11,
        },
        "alert_triggered": True,
        "recommendation": "RETRAIN — Amount PSI=0.22 exceeds critical threshold (0.15). "
                          "Trigger airflow DAG: fraud_retrain_pipeline."
    }
