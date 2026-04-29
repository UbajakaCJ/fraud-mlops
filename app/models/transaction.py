"""
TransactionLog — MongoDB document model.

Documents are plain dicts; this module
provides helper functions to build and parse them.
"""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class TransactionLog(BaseModel):
    """Pydantic model that mirrors a transaction_logs document."""
    model_config = ConfigDict(protected_namespaces=())

    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    model_name: str
    model_version: str
    latency_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo(self) -> dict:
        d = self.model_dump()
        return d

    @classmethod
    def from_mongo(cls, doc: dict) -> "TransactionLog":
        doc = dict(doc)
        doc.pop("_id", None)          # strip ObjectId
        return cls(**doc)
