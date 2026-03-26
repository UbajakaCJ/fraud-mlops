"""Audit Logger — writes prediction records asynchronously."""
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.transaction import TransactionLog

log = structlog.get_logger()


async def log_prediction(db: AsyncSession, response, transaction_features):
    try:
        record = TransactionLog(
            transaction_id=response.transaction_id,
            is_fraud=response.is_fraud,
            fraud_probability=response.fraud_probability,
            risk_level=response.risk_level.value,
            model_name=response.model_name,
            model_version=response.model_version,
            latency_ms=response.latency_ms,
        )
        db.add(record)
        await db.commit()
    except Exception as e:
        log.error("audit_log_failed", error=str(e))
