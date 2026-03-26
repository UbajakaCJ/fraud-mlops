from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime
from sqlalchemy.sql import func
from app.core.database import Base


class TransactionLog(Base):
    __tablename__ = "transaction_logs"

    id               = Column(Integer, primary_key=True, index=True)
    transaction_id   = Column(String, unique=True, index=True, nullable=False)
    is_fraud         = Column(Boolean, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    risk_level       = Column(String, nullable=False)
    model_name       = Column(String, nullable=False)
    model_version    = Column(String, nullable=False)
    latency_ms       = Column(Float)
    created_at       = Column(DateTime(timezone=True), server_default=func.now(), index=True)
