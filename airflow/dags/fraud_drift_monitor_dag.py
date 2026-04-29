"""
fraud_drift_monitor.py
──────────────────────
Airflow DAG: Hourly feature drift monitoring using PSI (Population Stability Index).

Triggers the retrain DAG automatically if drift exceeds critical threshold.
"""

from __future__ import annotations
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator

log = logging.getLogger(__name__)

MODEL_PATH = os.getenv("FRAUD_MODEL_PATH", "/opt/airflow/models")

PSI_WARNING  = 0.10
PSI_CRITICAL = 0.20


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """PSI = Σ (actual% - expected%) × ln(actual% / expected%). Kept inline for Airflow worker portability."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] = np.inf

    def bucket_pct(arr):
        counts, _ = np.histogram(arr, bins=breakpoints)
        pct = counts / len(arr)
        return np.where(pct == 0, 1e-6, pct)

    e_pct = bucket_pct(expected)
    a_pct = bucket_pct(actual)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

DEFAULT_ARGS = {
    "owner":            "chijioke.ubajaka",
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": True,
    "email":            ["cubajaka@bluechiptech.biz"],
}


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index: PSI = Σ (actual% - expected%) × ln(actual% / expected%)"""
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] = np.inf

    def bucket_pct(arr):
        counts, _ = np.histogram(arr, bins=breakpoints)
        pct = counts / len(arr)
        return np.where(pct == 0, 1e-6, pct)

    e_pct = bucket_pct(expected)
    a_pct = bucket_pct(actual)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def compute_drift(**ctx):
    """Compare current prediction window features against training baseline."""
    from pathlib import Path
    import joblib

    baseline_path = Path(MODEL_PATH) / "splits" / "X_train_eng.pkl"
    if not baseline_path.exists():
        log.warning("No baseline found — skipping drift check")
        ctx["ti"].xcom_push(key="max_psi", value=0.0)
        ctx["ti"].xcom_push(key="drift_results", value={})
        return

    X_baseline = joblib.load(baseline_path)
    # In production, X_current comes from your prediction log DB
    # Here we simulate with slight distribution shift
    rng = np.random.default_rng(int(datetime.now().timestamp()) % 10000)
    X_current = X_baseline.copy()
    X_current["Amount"]     = X_current.get("log_amount", X_current.iloc[:, 0]) * rng.uniform(0.8, 1.5)
    X_current["V14"]        = X_current.get("V14", X_current.iloc[:, 13]) + rng.normal(0.3, 0.5, len(X_current))

    features = X_baseline.columns.tolist()
    psi_scores = {}
    for feat in features:
        try:
            psi = compute_psi(X_baseline[feat].values, X_current[feat].values)
            psi_scores[feat] = round(psi, 4)
        except Exception as e:
            log.warning(f"PSI failed for {feat}: {e}")

    drifted = {f: p for f, p in psi_scores.items() if p >= PSI_WARNING}
    max_psi = max(psi_scores.values(), default=0.0)

    log.info(f"Drift check: max_psi={max_psi:.4f}, drifted features={list(drifted.keys())}")
    ctx["ti"].xcom_push(key="max_psi",       value=max_psi)
    ctx["ti"].xcom_push(key="psi_scores",    value=psi_scores)
    ctx["ti"].xcom_push(key="drift_results", value=drifted)


def decide_action(**ctx):
    max_psi = ctx["ti"].xcom_pull(task_ids="compute_drift", key="max_psi") or 0.0
    if max_psi >= PSI_CRITICAL:
        log.warning(f"CRITICAL DRIFT — PSI={max_psi:.4f} ≥ {PSI_CRITICAL} → triggering retrain")
        return "trigger_retrain"
    elif max_psi >= PSI_WARNING:
        log.info(f"WARNING drift — PSI={max_psi:.4f} — logging alert")
        return "log_drift_warning"
    return "no_action"


def log_drift_warning(**ctx):
    scores = ctx["ti"].xcom_pull(task_ids="compute_drift", key="drift_results") or {}
    log.warning(f"[DRIFT WARNING] Features above PSI {PSI_WARNING}: {scores}")


with DAG(
    dag_id="fraud_drift_monitor",
    description="Hourly PSI-based feature drift monitor — auto-triggers retrain on critical drift",
    default_args=DEFAULT_ARGS,
    schedule="0 * * * *",      # Hourly
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "monitoring", "drift"],
) as dag:

    t_drift = PythonOperator(task_id="compute_drift", python_callable=compute_drift)

    t_decide = BranchPythonOperator(task_id="decide_action", python_callable=decide_action)

    t_trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="fraud_retrain_pipeline",
        reset_dag_run=True,
        wait_for_completion=False,
        conf={"triggered_by": "drift_monitor"},
    )

    t_warn     = PythonOperator(task_id="log_drift_warning", python_callable=log_drift_warning)
    t_no_action = EmptyOperator(task_id="no_action")

    t_drift >> t_decide >> [t_trigger_retrain, t_warn, t_no_action]
