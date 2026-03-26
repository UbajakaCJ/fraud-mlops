"""Redis prediction cache."""
import json
import structlog
import redis.asyncio as aioredis
from typing import Optional
from app.core.config import settings

log = structlog.get_logger()
_redis = None


async def _get_redis():
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


async def get_cached_prediction(txn_id: str) -> Optional[dict]:
    try:
        r = await _get_redis()
        data = await r.get(f"pred:{txn_id}")
        return json.loads(data) if data else None
    except Exception as e:
        log.warning("cache_get_failed", error=str(e))
        return None


async def set_cached_prediction(txn_id: str, response) -> None:
    try:
        r = await _get_redis()
        await r.setex(
            f"pred:{txn_id}",
            settings.PREDICTION_CACHE_TTL,
            response.model_dump_json(),
        )
    except Exception as e:
        log.warning("cache_set_failed", error=str(e))
