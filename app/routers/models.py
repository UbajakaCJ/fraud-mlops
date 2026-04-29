"""
Model Registry Router
---------------------
GET    /models                        — list all registered models
GET    /models/{name}/versions        — list versions
GET    /models/champion               — current champion info
POST   /models/register               — register MLflow run as model
POST   /models/promote                — promote version to stage
POST   /models/reload                 — hot-swap champion without restart
DELETE /models/{name}/versions/{ver}  — archive a version
"""

import mlflow
import mlflow.sklearn
import structlog
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.core.config import settings
from app.core.model_registry import model_registry
from app.schemas.schemas import (
    ModelInfo, ModelPromoteRequest, ModelRegisterRequest, ModelStage
)

log = structlog.get_logger()
router = APIRouter()


def _get_client():
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()


def _parse_model_version(mv) -> ModelInfo:
    run = None
    metrics, params, tags = {}, {}, {}
    try:
        client = _get_client()
        run = client.get_run(mv.run_id)
        metrics = dict(run.data.metrics)
        params  = dict(run.data.params)
        tags    = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
    except Exception:
        pass
    return ModelInfo(
        name=mv.name,
        version=mv.version,
        stage=mv.current_stage,
        run_id=mv.run_id,
        metrics=metrics,
        params=params,
        tags=tags,
        created_at=datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc) if mv.creation_timestamp else None,
        description=mv.description,
    )


@router.get("/champion", summary="Get current champion model info")
async def get_champion():
    """Returns metadata for the currently loaded champion model."""
    return model_registry.info()


@router.get("/", response_model=List[dict], summary="List all registered models")
async def list_models():
    """List all models registered in the MLflow Model Registry."""
    try:
        client = _get_client()
        registered = client.search_registered_models()
        return [
            {
                "name": m.name,
                "latest_versions": [
                    {"version": v.version, "stage": v.current_stage, "run_id": v.run_id}
                    for v in m.latest_versions
                ],
                "description": m.description,
                "tags": dict(m.tags),
            }
            for m in registered
        ]
    except Exception as e:
        raise HTTPException(503, f"MLflow unavailable: {e}")


@router.get("/{model_name}/versions", response_model=List[ModelInfo],
            summary="List all versions of a model")
async def list_versions(model_name: str):
    """Returns all versions for a named model, with metrics and params."""
    try:
        client = _get_client()
        versions = client.search_model_versions(f"name='{model_name}'")
        return [_parse_model_version(v) for v in versions]
    except Exception as e:
        raise HTTPException(404, f"Model '{model_name}' not found: {e}")


@router.post("/register", summary="Register an MLflow run as a model version")
async def register_model(payload: ModelRegisterRequest):
    """
    Promote an MLflow training run into the Model Registry.

    Use this after a successful Airflow retraining run to create a new
    candidate version for review before promotion to Production.
    """
    try:
        client = _get_client()
        mv = mlflow.register_model(
            model_uri=f"runs:/{payload.run_id}/model",
            name=payload.model_name,
            tags=payload.tags or {},
        )
        if payload.description:
            client.update_model_version(
                name=payload.model_name,
                version=mv.version,
                description=payload.description,
            )
        log.info("model_registered", name=mv.name, version=mv.version, run_id=payload.run_id)
        return {"message": "Registered", "name": mv.name, "version": mv.version}
    except Exception as e:
        raise HTTPException(500, f"Registration failed: {e}")


@router.post("/promote", summary="Promote model version to a deployment stage")
async def promote_model(payload: ModelPromoteRequest):
    """
    Transition a model version to Staging, Production, or Archived.

    Promoting to **Production** triggers a live hot-swap on the next
    `/models/reload` call or at the next Airflow health check interval.

    A justification string is required for audit compliance.
    """
    try:
        client = _get_client()
        client.transition_model_version_stage(
            name=payload.model_name,
            version=payload.version,
            stage=payload.target_stage.value,
            archive_existing_versions=(payload.target_stage == ModelStage.PRODUCTION),
        )
        client.update_model_version(
            name=payload.model_name,
            version=payload.version,
            description=f"[{datetime.now(timezone.utc).isoformat()}] {payload.justification}",
        )
        log.info("model_promoted",
                 name=payload.model_name, version=payload.version,
                 stage=payload.target_stage, justification=payload.justification)
        return {
            "message": f"Promoted to {payload.target_stage}",
            "model_name": payload.model_name,
            "version": payload.version,
        }
    except Exception as e:
        raise HTTPException(500, f"Promotion failed: {e}")


@router.post("/reload", summary="Hot-swap the champion model without restart")
async def reload_champion():
    """
    Reload the champion model from MLflow without restarting the API.

    Call this after promoting a new version to Production.
    Zero-downtime: in-flight requests continue with old model; new requests
    use the reloaded model once the lock is released.
    """
    try:
        old = model_registry.champion_name
        await model_registry.load_champion()
        new = model_registry.champion_name
        log.info("champion_reloaded", old=old, new=new)
        return {
            "message": "Champion reloaded",
            "previous": old,
            "current": new,
            "version": model_registry.champion_version,
        }
    except Exception as e:
        raise HTTPException(500, f"Reload failed: {e}")


@router.delete("/{model_name}/versions/{version}", summary="Archive a model version")
async def archive_version(model_name: str, version: str):
    """Archive a model version (prevents future promotion but preserves history)."""
    try:
        client = _get_client()
        client.transition_model_version_stage(model_name, version, "Archived")
        return {"message": f"Archived {model_name} v{version}"}
    except Exception as e:
        raise HTTPException(500, f"Archive failed: {e}")
