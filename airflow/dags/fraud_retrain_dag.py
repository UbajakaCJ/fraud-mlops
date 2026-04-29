"""
fraud_retrain_pipeline.py
─────────────────────────
Airflow DAG: End-to-end fraud model retraining pipeline.

Schedule : Weekly (Sunday 02:00 Lagos time) + on-demand trigger
Owner    : Chijioke Ubajaka — Bluechip AI Services

DAG Flow:
  data_quality_check
       ↓
  extract_and_validate
       ↓
  feature_engineering
       ↓
  ┌────┴────┐
  train_lr  train_rf  train_xgb  train_lgbm   (parallel)
  └────┬────┘
  evaluate_and_compare
       ↓
  register_champion_candidate
       ↓
  run_integration_tests
       ↓
  promote_to_production          (only if tests pass)
       ↓
  notify_team
       ↓
  reload_api_champion
"""

from __future__ import annotations

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable

log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_PATH    = os.getenv("FRAUD_MODEL_PATH", "/opt/airflow/models")
EXPERIMENT    = "fraud-detection"
REGISTRY_NAME = "fraud-detector-champion"
API_BASE_URL  = "http://api:8000"

DEFAULT_ARGS = {
    "owner":            "chijioke.ubajaka",
    "depends_on_past":  False,
    "email":            ["cubajaka@bluechiptech.biz"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

AUPRC_THRESHOLD = float(Variable.get("fraud_auprc_threshold", default_var="0.80"))
SMOTE_RATIO     = float(Variable.get("fraud_smote_ratio",     default_var="0.10"))


# ─── Task functions ───────────────────────────────────────────────────────────

def data_quality_check(**ctx):
    """
    Assert data quality gates before training:
    - Min row count
    - Fraud rate in expected range
    - No feature nulls
    - Schema validation
    """
    log.info("Running data quality checks...")
    data_path = Variable.get("fraud_data_path", default_var=f"{MODEL_PATH}/creditcard.csv")

    if not Path(data_path).exists():
        log.warning("Real dataset not found — using synthetic mock for pipeline validation")
        _generate_mock_data(data_path)

    df = pd.read_csv(data_path)

    assert len(df) >= 1000,         f"Too few rows: {len(df)}"
    fraud_rate = df["Class"].mean()
    assert 0.0001 <= fraud_rate <= 0.5, f"Fraud rate out of range: {fraud_rate}"
    assert df.isnull().sum().sum() == 0, "Null values detected in dataset"
    expected_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    missing = set(expected_cols) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    ctx["ti"].xcom_push(key="data_path",    value=data_path)
    ctx["ti"].xcom_push(key="row_count",    value=len(df))
    ctx["ti"].xcom_push(key="fraud_rate",   value=float(fraud_rate))
    log.info(f"Data quality OK — rows={len(df)}, fraud_rate={fraud_rate:.4%}")


def _generate_mock_data(path: str, n: int = 10_000):
    """Generate synthetic creditcard dataset for CI/pipeline validation."""
    rng = np.random.default_rng(42)
    n_fraud = int(n * 0.0017)
    n_legit = n - n_fraud

    def block(size, fraud=False):
        base = rng.standard_normal((size, 28))
        if fraud:
            base[:, [0, 3, 9, 11, 13, 15]] += np.array([-3, -2, 2, -4, -3, -2])
        d = {f"V{i+1}": base[:, i] for i in range(28)}
        d["Time"]   = rng.uniform(0, 172800, size)
        d["Amount"] = rng.exponential(10 if fraud else 88, size)
        d["Class"]  = int(fraud)
        return pd.DataFrame(d)

    df = pd.concat([block(n_legit), block(n_fraud, True)]).sample(frac=1, random_state=42)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Mock data written to {path}")


def extract_and_validate(**ctx):
    """Load data, split train/val/test with stratification, push splits to XCom."""
    import joblib
    from sklearn.model_selection import train_test_split

    data_path = ctx["ti"].xcom_pull(key="data_path")
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.15, stratify=y_tv, random_state=42)

    split_dir = Path(MODEL_PATH) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
                      ("y_train", y_train), ("y_val",  y_val), ("y_test",  y_test)]:
        joblib.dump(arr, split_dir / f"{name}.pkl")

    ctx["ti"].xcom_push(key="split_dir",    value=str(split_dir))
    ctx["ti"].xcom_push(key="train_size",   value=len(X_train))
    ctx["ti"].xcom_push(key="val_size",     value=len(X_val))
    ctx["ti"].xcom_push(key="test_size",    value=len(X_test))
    log.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")


def feature_engineering(**ctx):
    """Apply feature transforms — must mirror app/services/feature_engineering.py exactly."""
    import joblib

    split_dir = Path(ctx["ti"].xcom_pull(key="split_dir"))

    for split in ("train", "val", "test"):
        X = joblib.load(split_dir / f"X_{split}.pkl")
        X = X.copy()
        X["log_amount"] = np.log1p(X["Amount"])
        X["time_hour"]  = (X["Time"] % 86400) / 3600
        X["time_sin"]   = np.sin(2 * np.pi * X["time_hour"] / 24)
        X["time_cos"]   = np.cos(2 * np.pi * X["time_hour"] / 24)
        X = X.drop(columns=["Time", "Amount"])
        joblib.dump(X, split_dir / f"X_{split}_eng.pkl")

    ctx["ti"].xcom_push(key="split_dir", value=str(split_dir))
    log.info("Feature engineering complete")


def _train_model(model_type: str, **ctx):
    """Generic trainer — called by each model-specific task."""
    import mlflow
    import mlflow.sklearn
    import joblib
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    import lightgbm as lgb

    split_dir = Path(ctx["ti"].xcom_pull(task_ids="feature_engineering", key="split_dir"))

    X_train = joblib.load(split_dir / "X_train_eng.pkl")
    y_train = joblib.load(split_dir / "y_train.pkl")
    X_val   = joblib.load(split_dir / "X_val_eng.pkl")
    y_val   = joblib.load(split_dir / "y_val.pkl")

    model_configs = {
        "logistic_regression": {
            "clf": LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1, random_state=42),
            "params": {"C": 0.1, "solver": "lbfgs"},
        },
        "random_forest": {
            "clf": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                          max_depth=12, n_jobs=-1, random_state=42),
            "params": {"n_estimators": 200, "max_depth": 12},
        },
        "xgboost": {
            "clf": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     eval_metric="aucpr", use_label_encoder=False,
                                     random_state=42, n_jobs=-1, verbosity=0),
            "params": {"n_estimators": 300, "max_depth": 6, "lr": 0.05},
        },
        "lightgbm": {
            "clf": lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      class_weight="balanced", random_state=42,
                                      n_jobs=-1, verbose=-1),
            "params": {"n_estimators": 300, "max_depth": 6, "lr": 0.05},
        },
    }

    cfg = model_configs[model_type]
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42, sampling_strategy=SMOTE_RATIO)),
        ("scaler", StandardScaler()),
        ("clf", cfg["clf"]),
    ]) if model_type == "logistic_regression" else ImbPipeline([
        ("smote", SMOTE(random_state=42, sampling_strategy=SMOTE_RATIO)),
        ("clf", cfg["clf"]),
    ])

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=f"{model_type}-{ctx['ds_nodash']}") as run:
        mlflow.set_tags({
            "model_type": model_type,
            "airflow_dag": ctx["dag"].dag_id,
            "airflow_run": ctx["run_id"],
            "env": "training",
        })
        mlflow.log_params({**cfg["params"], "smote_ratio": SMOTE_RATIO})

        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_val)[:, 1]
        y_pred = pipeline.predict(X_val)

        metrics = {
            "auprc":     average_precision_score(y_val, y_prob),
            "auroc":     roc_auc_score(y_val, y_prob),
            "f1":        f1_score(y_val, y_pred),
            "val_fraud": int(y_val.sum()),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model", registered_model_name=None)

        run_id = run.info.run_id

    ctx["ti"].xcom_push(key=f"{model_type}_run_id",  value=run_id)
    ctx["ti"].xcom_push(key=f"{model_type}_auprc",   value=metrics["auprc"])
    log.info(f"{model_type} trained — run_id={run_id}, AUPRC={metrics['auprc']:.4f}")


def train_logistic_regression(**ctx): _train_model("logistic_regression", **ctx)
def train_random_forest(**ctx):       _train_model("random_forest", **ctx)
def train_xgboost(**ctx):             _train_model("xgboost", **ctx)
def train_lightgbm(**ctx):            _train_model("lightgbm", **ctx)


def evaluate_and_compare(**ctx):
    """Pick the best model by AUPRC on the validation set."""
    ti = ctx["ti"]
    model_types = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

    scores = {}
    for m in model_types:
        auprc = ti.xcom_pull(task_ids=f"train_{m}", key=f"{m}_auprc")
        if auprc is not None:
            scores[m] = auprc

    if not scores:
        raise ValueError("No model scores found in XCom")

    best = max(scores, key=scores.get)
    best_auprc = scores[best]
    best_run_id = ti.xcom_pull(task_ids=f"train_{best}", key=f"{best}_run_id")

    log.info(f"Best model: {best} (AUPRC={best_auprc:.4f})")
    log.info(f"All scores: {scores}")

    ti.xcom_push(key="best_model_type",  value=best)
    ti.xcom_push(key="best_run_id",      value=best_run_id)
    ti.xcom_push(key="best_auprc",       value=best_auprc)
    ti.xcom_push(key="all_scores",       value=json.dumps(scores))

    if best_auprc < AUPRC_THRESHOLD:
        log.warning(f"Best AUPRC {best_auprc:.4f} below threshold {AUPRC_THRESHOLD} — will not promote")


def register_champion_candidate(**ctx):
    """Register the best run in the MLflow Model Registry as a Staging candidate."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    ti = ctx["ti"]
    run_id    = ti.xcom_pull(task_ids="evaluate_and_compare", key="best_run_id")
    model_type = ti.xcom_pull(task_ids="evaluate_and_compare", key="best_model_type")
    auprc      = ti.xcom_pull(task_ids="evaluate_and_compare", key="best_auprc")

    mv = mlflow.register_model(f"runs:/{run_id}/model", REGISTRY_NAME)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(REGISTRY_NAME, mv.version, "Staging")
    client.update_model_version(
        name=REGISTRY_NAME, version=mv.version,
        description=f"Candidate: {model_type} | AUPRC={auprc:.4f} | DAG run={ctx['run_id']}"
    )
    ti.xcom_push(key="candidate_version", value=mv.version)
    log.info(f"Registered v{mv.version} to Staging — {model_type}, AUPRC={auprc:.4f}")


def run_integration_tests(**ctx):
    """
    Run smoke tests against the Staging model before production promotion.
    Tests: prediction shape, output range, latency SLA, known fraud case.
    """
    import joblib
    import mlflow.sklearn
    mlflow.set_tracking_uri(MLFLOW_URI)

    ti = ctx["ti"]
    split_dir = Path(ti.xcom_pull(task_ids="feature_engineering", key="split_dir"))
    version   = ti.xcom_pull(task_ids="register_champion_candidate", key="candidate_version")

    model = mlflow.sklearn.load_model(f"models:/{REGISTRY_NAME}/{version}")
    X_test = joblib.load(split_dir / "X_test_eng.pkl")
    y_test = joblib.load(split_dir / "y_test.pkl")

    import time
    from sklearn.metrics import average_precision_score

    # Test 1: predict runs
    t0 = time.perf_counter()
    probs = model.predict_proba(X_test)[:, 1]
    latency = (time.perf_counter() - t0) * 1000

    assert probs.shape[0] == len(X_test), "Output length mismatch"
    assert probs.min() >= 0.0 and probs.max() <= 1.0, "Probabilities out of [0,1]"

    # Test 2: latency SLA (< 5s for full test set)
    assert latency < 5000, f"Latency SLA failed: {latency:.0f}ms"

    # Test 3: AUPRC on holdout
    auprc = average_precision_score(y_test, probs)
    assert auprc >= AUPRC_THRESHOLD, f"AUPRC {auprc:.4f} below threshold {AUPRC_THRESHOLD}"

    ti.xcom_push(key="test_auprc",   value=auprc)
    ti.xcom_push(key="test_latency", value=latency)
    log.info(f"Integration tests passed — AUPRC={auprc:.4f}, latency={latency:.0f}ms")


def should_promote(**ctx):
    """Branch: promote to production only if integration tests passed."""
    ti = ctx["ti"]
    test_auprc = ti.xcom_pull(task_ids="run_integration_tests", key="test_auprc")
    if test_auprc is not None and test_auprc >= AUPRC_THRESHOLD:
        return "promote_to_production"
    return "notify_failure"


def promote_to_production(**ctx):
    """Promote Staging model to Production, archive old Production."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    ti      = ctx["ti"]
    version = ti.xcom_pull(task_ids="register_champion_candidate", key="candidate_version")
    auprc   = ti.xcom_pull(task_ids="run_integration_tests",       key="test_auprc")
    mtype   = ti.xcom_pull(task_ids="evaluate_and_compare",        key="best_model_type")

    client.transition_model_version_stage(
        REGISTRY_NAME, version, "Production",
        archive_existing_versions=True,
    )
    client.update_model_version(
        name=REGISTRY_NAME, version=version,
        description=f"PROMOTED: {mtype} | test AUPRC={auprc:.4f} | {ctx['ds']}"
    )
    log.info(f"Promoted v{version} to Production — {mtype}, AUPRC={auprc:.4f}")


def reload_api_champion(**ctx):
    """Call the FastAPI /models/reload endpoint to hot-swap the champion."""
    import requests
    try:
        resp = requests.post(f"{API_BASE_URL}/models/reload", timeout=30)
        resp.raise_for_status()
        log.info(f"API champion reloaded: {resp.json()}")
    except Exception as e:
        log.error(f"API reload failed (manual reload required): {e}")


def notify_team(**ctx):
    """Send success notification (Slack/email in production)."""
    ti = ctx["ti"]
    auprc   = ti.xcom_pull(task_ids="run_integration_tests", key="test_auprc")
    version = ti.xcom_pull(task_ids="register_champion_candidate", key="candidate_version")
    mtype   = ti.xcom_pull(task_ids="evaluate_and_compare", key="best_model_type")
    all_scores = ti.xcom_pull(task_ids="evaluate_and_compare", key="all_scores")
    log.info(
        f"[NOTIFY] Retraining complete.\n"
        f"  Champion: {mtype} v{version}\n"
        f"  AUPRC:    {auprc:.4f}\n"
        f"  All scores: {all_scores}\n"
        f"  MLflow:   {MLFLOW_URI}/#/models/{REGISTRY_NAME}\n"
    )


def notify_failure(**ctx):
    log.error("[NOTIFY] Model failed integration tests — NOT promoted to production. Manual review required.")


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="fraud_retrain_pipeline",
    description="End-to-end fraud model retraining: ingest → train → evaluate → register → promote",
    default_args=DEFAULT_ARGS,
    schedule="0 2 * * 0",   # Weekly, Sunday 02:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "mlops", "production"],
    doc_md=__doc__,
) as dag:

    t_data_quality = PythonOperator(
        task_id="data_quality_check",
        python_callable=data_quality_check,
        doc_md="Assert dataset quality gates (row count, fraud rate, nulls, schema).",
    )

    t_extract = PythonOperator(
        task_id="extract_and_validate",
        python_callable=extract_and_validate,
        doc_md="Stratified train/val/test split.",
    )

    t_features = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
        doc_md="Apply log-amount and cyclical time transforms.",
    )

    t_train_lr  = PythonOperator(task_id="train_logistic_regression", python_callable=train_logistic_regression)
    t_train_rf  = PythonOperator(task_id="train_random_forest",        python_callable=train_random_forest)
    t_train_xgb = PythonOperator(task_id="train_xgboost",              python_callable=train_xgboost)
    t_train_lgb = PythonOperator(task_id="train_lightgbm",             python_callable=train_lightgbm)

    t_evaluate = PythonOperator(
        task_id="evaluate_and_compare",
        python_callable=evaluate_and_compare,
        doc_md="Pick champion by AUPRC on validation set.",
    )

    t_register = PythonOperator(
        task_id="register_champion_candidate",
        python_callable=register_champion_candidate,
        doc_md="Register best run to MLflow Registry as Staging.",
    )

    t_integration = PythonOperator(
        task_id="run_integration_tests",
        python_callable=run_integration_tests,
        doc_md="Smoke tests: shape, probability range, latency SLA, AUPRC holdout.",
    )

    t_branch = BranchPythonOperator(
        task_id="should_promote",
        python_callable=should_promote,
    )

    t_promote = PythonOperator(
        task_id="promote_to_production",
        python_callable=promote_to_production,
    )

    t_notify_fail = PythonOperator(
        task_id="notify_failure",
        python_callable=notify_failure,
    )

    t_notify = PythonOperator(
        task_id="notify_team",
        python_callable=notify_team,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    t_reload = PythonOperator(
        task_id="reload_api_champion",
        python_callable=reload_api_champion,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ─── Dependency graph ──────────────────────────────────────────────────────
    (
        t_data_quality
        >> t_extract
        >> t_features
        >> [t_train_lr, t_train_rf, t_train_xgb, t_train_lgb]
        >> t_evaluate
        >> t_register
        >> t_integration
        >> t_branch
    )

    t_branch >> t_promote       >> [t_notify, t_reload]
    t_branch >> t_notify_fail   >> t_notify