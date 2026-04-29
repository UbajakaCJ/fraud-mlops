"""
MongoDB async database 

Provides:
  - A Motor AsyncIOMotorClient singleton
  - get_motor_db() — returns the frauddb database handle
  - get_db()       — FastAPI dependency (mirrors old interface)
  - init_db() / close_db() — lifecycle hooks called from main.py
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings
import structlog

log = structlog.get_logger()

_client: AsyncIOMotorClient | None = None


async def init_db() -> None:
    global _client
    _client = AsyncIOMotorClient(settings.MONGODB_URI)
    _db = _client[settings.MONGODB_DB_NAME]
    col = _db["transaction_logs"]
    await col.create_index("transaction_id", unique=True)
    await col.create_index("created_at")
    await col.create_index("is_fraud")
    log.info("mongodb_connected", db=settings.MONGODB_DB_NAME)


async def close_db() -> None:
    global _client
    if _client:
        _client.close()
        log.info("mongodb_disconnected")


def get_motor_db() -> AsyncIOMotorDatabase:
    if _client is None:
        raise RuntimeError("MongoDB client not initialised — call init_db() first")
    return _client[settings.MONGODB_DB_NAME]


async def get_db():
    """FastAPI dependency — yields Motor database."""
    yield get_motor_db()
