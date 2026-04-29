"""
Model Registry
--------------
Loads the champion model from MLflow Model Registry.
Falls back to local .pkl if MLflow is unavailable (cold-start resilience).
Supports hot-swap without restart via /models/reload endpoint.
"""

import os
import asyncio
import joblib
import numpy as np
import structlog
from pathlib import Path
from typing import Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

log = structlog.get_logger()


class ModelRegistry:
    def __init__(self):
        self.champion: Optional[Any] = None
        self.challenger: Optional[Any] = None
        self.champion_name: str = "unknown"
        self.champion_version: str = "unknown"
        self.champion_run_id: str = "unknown"
        self._lock = asyncio.Lock()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def load_champion(self) -> None:
        async with self._lock:
            await asyncio.get_event_loop().run_in_executor(None, self._load_sync)

    def _load_sync(self):
        import mlflow
        import mlflow.sklearn
        import warnings
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        # Try MLflow Model Registry — use search_model_versions (non-deprecated)
        # Load by run_id URI to avoid the deprecated stage-based get_latest_versions
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{settings.CHAMPION_MODEL_NAME}'")
            production = [v for v in versions if v.current_stage == "Production"]

            if production:
                mv = production[0]
                # Load by run URI (avoids deprecated stage-based loading path in MLflow internals)
                model_uri = f"runs:/{mv.run_id}/model"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    self.champion = mlflow.sklearn.load_model(model_uri)
                self.champion_name    = mv.name
                self.champion_version = mv.version
                self.champion_run_id  = mv.run_id
                log.info("model_loaded_from_registry",
                         name=mv.name, version=mv.version, run_id=mv.run_id)
                return
        except Exception as e:
            log.warning("mlflow_registry_unavailable", error=str(e))

        # Fallback: local .pkl — picks the most recently modified file
        model_path = Path(settings.MODEL_PATH)
        local_paths = sorted(
            model_path.glob("*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if local_paths:
            path = local_paths[0]
            self.champion = joblib.load(path)
            self.champion_name    = path.stem
            self.champion_version = "local"
            self.champion_run_id  = "local"
            log.info("model_loaded_from_disk", path=str(path))
            return

        # No model anywhere — start in degraded mode
        log.warning("no_model_found_degraded_mode",
                    model_path=str(model_path),
                    hint="Copy a .pkl file to the model/ directory or run the Airflow retrain DAG")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (predictions, probabilities)."""
        if self.champion is None:
            raise RuntimeError("Model not loaded — copy a .pkl to model/ or trigger /models/reload")
        probs = self.champion.predict_proba(X)[:, 1]
        preds = (probs >= settings.PREDICTION_THRESHOLD).astype(int)
        return preds, probs

    def is_ready(self) -> bool:
        return self.champion is not None

    def info(self) -> dict:
        return {
            "name":      self.champion_name,
            "version":   self.champion_version,
            "run_id":    self.champion_run_id,
            "threshold": settings.PREDICTION_THRESHOLD,
            "ready":     self.is_ready(),
        }


model_registry = ModelRegistry()