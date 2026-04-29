"""
Experiments Router
------------------
GET /experiments                    — list all experiments
GET /experiments/runs/{run_id}      — single run detail  [STATIC - must be before dynamic]
GET /experiments/compare/runs       — side-by-side comparison [STATIC - must be before dynamic]
GET /experiments/{name}/runs        — list runs with metrics  [DYNAMIC - last]
"""

import mlflow
import structlog
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from app.core.config import settings

log = structlog.get_logger()
router = APIRouter()


def _get_client():
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()


def _parse_run(run) -> dict:
    start = datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc) if run.info.start_time else None
    end   = datetime.fromtimestamp(run.info.end_time   / 1000, tz=timezone.utc) if run.info.end_time   else None
    return {
        "run_id":       run.info.run_id,
        "run_name":     run.info.run_name,
        "status":       run.info.status,
        "start_time":   start,
        "end_time":     end,
        "duration_s":   round((run.info.end_time - run.info.start_time) / 1000, 1) if run.info.end_time else None,
        "metrics":      dict(run.data.metrics),
        "params":       dict(run.data.params),
        "tags":         {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
        "artifact_uri": run.info.artifact_uri,
    }


@router.get("/", summary="List all MLflow experiments")
async def list_experiments():
    """Returns all experiments in the MLflow tracking server."""
    try:
        client = _get_client()
        exps = client.search_experiments()
        return [
            {
                "experiment_id":   e.experiment_id,
                "name":            e.name,
                "lifecycle_stage": e.lifecycle_stage,
                "artifact_location": e.artifact_location,
                "tags": dict(e.tags),
            }
            for e in exps
        ]
    except Exception as e:
        raise HTTPException(503, f"MLflow unavailable: {e}")


# ── STATIC routes MUST come before dynamic /{experiment_name}/runs ──────────

@router.get("/runs/{run_id}", summary="Get a single run's full details")
async def get_run(run_id: str):
    """Returns the full run record including all metrics, params, and tags."""
    try:
        client = _get_client()
        run = client.get_run(run_id)
        return _parse_run(run)
    except Exception as e:
        raise HTTPException(404, f"Run '{run_id}' not found: {e}")


@router.get("/compare/runs", summary="Side-by-side comparison of multiple runs")
async def compare_runs(run_ids: List[str] = Query(..., description="Comma-separated run IDs")):
    """
    Returns a comparison table of metrics and params across multiple runs.

    Useful for champion/challenger analysis before promotion decisions.
    """
    try:
        client = _get_client()
        parsed = [_parse_run(client.get_run(rid)) for rid in run_ids]
        all_metrics = sorted({k for r in parsed for k in r["metrics"]})
        all_params  = sorted({k for r in parsed for k in r["params"]})
        return {
            "runs": parsed,
            "metric_comparison": {
                m: {r["run_id"]: r["metrics"].get(m) for r in parsed}
                for m in all_metrics
            },
            "param_comparison": {
                p: {r["run_id"]: r["params"].get(p) for r in parsed}
                for p in all_params
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── DYNAMIC route last ───────────────────────────────────────────────────────

@router.get("/{experiment_name}/runs", summary="List runs for an experiment")
async def list_runs(
    experiment_name: str,
    max_results: int = Query(50, le=500),
    order_by: str = Query("metrics.auprc DESC", description="MLflow order_by clause"),
    filter_string: str = Query("", description="MLflow filter string e.g. 'metrics.auprc > 0.85'"),
):
    """
    Returns all runs for the named experiment, sorted and filtered.

    **Example filter_string values:**
    - `metrics.auprc > 0.85`
    - `params.model_type = 'XGBoost'`
    - `tags.env = 'production'`
    """
    try:
        client = _get_client()
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            raise HTTPException(404, f"Experiment '{experiment_name}' not found")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=filter_string,
            order_by=[order_by],
            max_results=max_results,
        )
        return {
            "experiment_id":   exp.experiment_id,
            "experiment_name": exp.name,
            "total_runs":      len(runs),
            "runs":            [_parse_run(r) for r in runs],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))