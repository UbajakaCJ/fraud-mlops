"""
Health Router
"""
import time
import redis.asyncio as aioredis
import structlog
from datetime import datetime, timezone
from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.model_registry import model_registry
from app.schemas.schemas import HealthResponse, HealthStatus, DependencyHealth

log = structlog.get_logger()
router = APIRouter()


async def _check_postgres():
    t0 = time.perf_counter()
    try:
        async with AsyncSessionLocal() as s:
            await s.execute(text("SELECT 1"))
        return DependencyHealth(name="postgres", status=HealthStatus.OK,
                                latency_ms=round((time.perf_counter()-t0)*1000, 2), detail=None)
    except Exception as e:
        return DependencyHealth(name="postgres", status=HealthStatus.DOWN, latency_ms=None, detail=str(e))


async def _check_redis():
    t0 = time.perf_counter()
    try:
        r = aioredis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.aclose()
        return DependencyHealth(name="redis", status=HealthStatus.OK,
                                latency_ms=round((time.perf_counter()-t0)*1000, 2), detail=None)
    except Exception as e:
        return DependencyHealth(name="redis", status=HealthStatus.DEGRADED, latency_ms=None, detail=str(e))


async def _check_mlflow():
    import httpx
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{settings.MLFLOW_TRACKING_URI}/health")
        return DependencyHealth(name="mlflow", status=HealthStatus.OK if resp.status_code == 200 else HealthStatus.DEGRADED,
                                latency_ms=round((time.perf_counter()-t0)*1000, 2), detail=None)
    except Exception as e:
        return DependencyHealth(name="mlflow", status=HealthStatus.DEGRADED, latency_ms=None, detail=str(e))


@router.get("/health", response_model=HealthResponse, summary="Full dependency health check",
            tags=["health"])
async def health():
    """Comprehensive health check: API, model, Postgres, Redis, MLflow."""
    deps = [await _check_postgres(), await _check_redis(), await _check_mlflow()]
    statuses = [d.status for d in deps]
    if any(s == HealthStatus.DOWN for s in statuses):
        overall = HealthStatus.DOWN
    elif any(s == HealthStatus.DEGRADED for s in statuses):
        overall = HealthStatus.DEGRADED
    else:
        overall = HealthStatus.OK

    return HealthResponse(
        status=overall,
        version=settings.API_VERSION,
        environment=settings.ENVIRONMENT,
        model=model_registry.info(),
        dependencies=deps,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/", tags=["health"], include_in_schema=False)
async def root():
    return {"service": "Fraud Detection API", "version": settings.API_VERSION, "docs": "/docs"}
