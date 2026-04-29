"""
Fraud Detection MLOps API
Bluechip Technologies — AI Services Department
"""

import warnings
warnings.filterwarnings("ignore", message=".*protected namespace.*")

from contextlib import asynccontextmanager
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import settings
from app.core.database import init_db, close_db
from app.core.model_registry import model_registry
from app.middleware.logging import LoggingMiddleware
from app.routers import predict, models, experiments, health, transactions, auth


log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    log.info("startup", env=settings.ENVIRONMENT, version=settings.API_VERSION)

    # Connect to MongoDB and create indexes
    await init_db()
    log.info("database_ready")

    # Load champion model from MLflow / local fallback
    await model_registry.load_champion()
    log.info("model_loaded", model=model_registry.champion_name)

    yield

    # Graceful shutdown
    await close_db()
    log.info("shutdown")


# ─── App definition ────────────────────────────────────────────────────────────

app = FastAPI(
    title="🛡️ Fraud Detection MLOps API",
    description="""
## Bluechip Technologies — Credit Card Fraud Detection Platform

A production-grade MLOps API serving real-time fraud predictions backed by an
automated retraining pipeline (Airflow), experiment tracking (MLflow),
and a full CI/CD release process.

### Capabilities
- **Real-time prediction** — single transaction scoring with confidence intervals
- **Batch prediction** — bulk scoring up to 10,000 transactions
- **Model management** — register, promote, champion/challenger rollouts
- **Experiment tracking** — full MLflow experiment lineage
- **Drift detection** — feature drift monitoring with alerts
- **Audit trail** — every prediction logged with full reproducibility

### Authentication
Bearer token (JWT). Use `POST /auth/token` with credentials to obtain a token.

### Rate Limits
- Free tier: 100 req/min
- Enterprise: 10,000 req/min
""",
    version=settings.API_VERSION,
    contact={
        "name": "Chijioke Ubajaka — AI Services",
        "email": "cubajaka@bluechiptech.biz",
        "url": "https://bluechiptech.biz/",
    },
    license_info={"name": "Proprietary — Bluechip Technologies"},
    openapi_tags=[
        {"name": "health",        "description": "Liveness, readiness, and dependency health checks"},
        {"name": "auth",          "description": "JWT authentication"},
        {"name": "predict",       "description": "Real-time and batch fraud prediction"},
        {"name": "transactions",  "description": "Prediction audit log and transaction history"},
        {"name": "models",        "description": "Model registry — register, promote, rollback"},
        {"name": "experiments",   "description": "MLflow experiment tracking and comparison"},
    ],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ─── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LoggingMiddleware)

# ─── Prometheus metrics ───────────────────────────────────────────────────────

Instrumentator(
    should_group_status_codes=True,
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app, include_in_schema=False)

# ─── Routers ─────────────────────────────────────────────────────────────────

app.include_router(health.router,       tags=["health"])
app.include_router(auth.router,         prefix="/auth",         tags=["auth"])
app.include_router(predict.router,      prefix="/predict",      tags=["predict"])
app.include_router(transactions.router, prefix="/transactions",  tags=["transactions"])
app.include_router(models.router,       prefix="/models",        tags=["models"])
app.include_router(experiments.router,  prefix="/experiments",   tags=["experiments"])


# ─── Global exception handler ────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ─── Custom OpenAPI schema (adds Authorize button) ───────────────────────────

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi
