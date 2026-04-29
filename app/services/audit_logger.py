"""Audit Logger — writes prediction records asynchronously to MongoDB."""
import structlog
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models.transaction import TransactionLog

log = structlog.get_logger()


async def log_prediction(db: AsyncIOMotorDatabase, response, transaction_features):
    try:
        record = TransactionLog(
            transaction_id=response.transaction_id,
            is_fraud=response.is_fraud,
            fraud_probability=response.fraud_probability,
            risk_level=response.risk_level.value,
            model_name=response.model_name,
            model_version=response.model_version,
            latency_ms=response.latency_ms,
            created_at=datetime.now(timezone.utc),
        )
        await db["transaction_logs"].insert_one(record.to_mongo())
    except Exception as e:
        log.error("audit_log_failed", error=str(e))
