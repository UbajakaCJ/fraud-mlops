# 🛡️ Fraud Detection MLOps Platform

A production-grade, end-to-end MLOps platform for real-time credit card fraud detection.

```
FastAPI  ·  MLflow  ·  Airflow  ·  Docker  ·  PostgreSQL  ·  Redis  ·  Nginx  ·  GitHub Actions
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions CI/CD                         │
│   lint → test → dag-validate → docker-build → deploy-staging/prod  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│                     Docker Compose Stack                            │
│                                                                     │
│  ┌──────────┐    ┌──────────────────┐    ┌────────────────────┐    │
│  │  Nginx   │───▶│   FastAPI (API)  │───▶│  MLflow Tracking   │    │
│  │  :80     │    │   :8000          │    │  :5000             │    │
│  └──────────┘    │                  │    └────────────────────┘    │
│                  │  /predict        │              │                │
│                  │  /predict/batch  │    ┌─────────▼──────────┐    │
│                  │  /models         │    │     PostgreSQL      │    │
│                  │  /experiments    │    │  airflow + frauddb  │    │
│                  │  /health         │    └────────────────────┘    │
│                  │  /docs (Swagger) │                               │
│                  └────────┬─────────┘    ┌────────────────────┐    │
│                           │              │       Redis         │    │
│                           │              │  Prediction cache   │    │
│                           │              └────────────────────┘    │
│  ┌────────────────────────▼──────────────────────────────────┐     │
│  │              Airflow (Scheduler + Webserver :8080)        │     │
│  │                                                           │     │
│  │  DAG 1: fraud_retrain_pipeline  (weekly + on-demand)     │     │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐  │     │
│  │  │data_check│─▶│ extract  │─▶│  feature_engineering   │  │     │
│  │  └──────────┘  └──────────┘  └──────────┬─────────────┘  │     │
│  │                                          │                │     │
│  │                          ┌───────────────┼──────────────┐ │     │
│  │                          ▼               ▼              ▼ │     │
│  │                      train_lr       train_xgb      train_lgb    │
│  │                          └───────────────┼──────────────┘ │     │
│  │                                          ▼                │     │
│  │                               evaluate_and_compare        │     │
│  │                                          ▼                │     │
│  │                              register_champion_candidate  │     │
│  │                                          ▼                │     │
│  │                               run_integration_tests       │     │
│  │                                          ▼                │     │
│  │                              ┌───────────┴──────────────┐ │     │
│  │                              ▼                          ▼ │     │
│  │                       promote_to_prod          notify_failure   │
│  │                              ▼                                  │
│  │                       reload_api_champion                │     │
│  │                                                           │     │
│  │  DAG 2: fraud_drift_monitor    (hourly)                  │     │
│  │  compute_drift → decide_action → trigger_retrain/warn    │     │
│  └───────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Docker ≥ 24 + Docker Compose v2
- 8 GB RAM minimum (16 GB recommended)
- Git

### 1. Clone & configure

```bash
git clone https://github.com/Bluechip-AI/fraud-mlops.git
cd fraud-mlops
cp .env.example .env
# Edit .env — set SECRET_KEY, DB passwords, etc.
```

### 2. (Optional) Add real dataset

```bash
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
cp ~/Downloads/creditcard.csv model/creditcard.csv
```

Without the CSV, the system generates a synthetic mock automatically.

### 3. Launch the full stack

```bash
docker compose up -d
```

### 4. Verify services

| Service        | URL                          | Credentials     |
|---------------|------------------------------|-----------------|
| **Swagger UI** | http://localhost/docs        | —               |
| **ReDoc**      | http://localhost/redoc       | —               |
| **MLflow UI**  | http://localhost:5000        | —               |
| **Airflow UI** | http://localhost:8080        | admin / admin   |
| **Health**     | http://localhost/health      | —               |
| **Metrics**    | http://localhost/metrics     | —               |

---

## API Endpoints

### Prediction

```bash
# Single transaction
curl -X POST http://localhost/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-001",
    "transaction": {
      "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
      "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
      "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
      "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
      "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
      "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
      "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
      "Amount": 149.62, "Time": 0.0
    }
  }'
```

Response:
```json
{
  "transaction_id": "TXN-001",
  "is_fraud": true,
  "fraud_probability": 0.847312,
  "risk_level": "HIGH",
  "confidence": "very_high",
  "threshold_used": 0.33,
  "model_name": "fraud-detector-champion",
  "model_version": "7",
  "latency_ms": 4.2,
  "timestamp": "2024-09-01T14:23:11Z"
}
```

```bash
# Batch scoring
curl -X POST http://localhost/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...]}'

# Feature drift report
curl http://localhost/predict/drift

# Champion model info
curl http://localhost/models/champion

# Hot-swap after new model promoted in MLflow
curl -X POST http://localhost/models/reload
```

### MLflow Experiments

```bash
# List experiments
curl http://localhost/experiments/

# Runs for an experiment (sorted by AUPRC)
curl "http://localhost/experiments/fraud-detection/runs?order_by=metrics.auprc+DESC"

# Compare runs
curl "http://localhost/experiments/compare/runs?run_ids=abc123&run_ids=def456"
```

### Model Registry

```bash
# Promote a version to Production
curl -X POST http://localhost/models/promote \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "fraud-detector-champion",
    "version": "7",
    "target_stage": "Production",
    "justification": "AUPRC 0.901 on holdout, passed all integration tests"
  }'
```

---

## MLOps Workflow

### Automated Retraining

The `fraud_retrain_pipeline` DAG runs every Sunday at 02:00 WAT and:

1. Validates data quality (schema, null check, fraud rate bounds)
2. Splits data stratified by class
3. Engineers features (log-amount, cyclical time)
4. Trains 4 models in parallel (LR, RF, XGBoost, LightGBM) with SMOTE
5. Selects champion by AUPRC on validation set
6. Registers candidate to MLflow Model Registry (Staging)
7. Runs integration tests (AUPRC threshold, latency SLA)
8. Promotes to Production and calls `/models/reload`

### Drift Monitoring

The `fraud_drift_monitor` DAG runs hourly, computing PSI scores for all features:

| PSI Range | Action |
|-----------|--------|
| < 0.10    | No action |
| 0.10–0.20 | Warning logged |
| ≥ 0.20    | Auto-trigger retrain |

### Manual Trigger

```bash
# Trigger retrain via Airflow CLI
docker compose exec airflow-scheduler \
  airflow dags trigger fraud_retrain_pipeline

# Or via Airflow REST API
curl -X POST http://localhost:8080/api/v1/dags/fraud_retrain_pipeline/dagRuns \
  -u admin:admin \
  -H "Content-Type: application/json" \
  -d '{"conf": {"triggered_by": "manual"}}'
```

---

## Development

```bash
# Run tests
pip install -r requirements.txt
pytest tests/ -v --cov=app

# Local API (no Docker)
uvicorn app.main:app --reload

# Lint
ruff check app/ tests/
ruff format app/ tests/
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | *(required)* | JWT signing key (≥32 chars) |
| `DATABASE_URL` | postgres://... | Async SQLAlchemy URL |
| `MLFLOW_TRACKING_URI` | http://mlflow:5000 | MLflow server |
| `REDIS_URL` | redis://redis:6379/0 | Cache |
| `MODEL_PATH` | ./model | Local model fallback |
| `PREDICTION_THRESHOLD` | 0.33 | Fraud decision threshold |
| `ENVIRONMENT` | development | development/production |

---

## CI/CD Pipeline

```
Push to develop ──▶ lint ──▶ test (py3.10, py3.11) ──▶ dag-validate
                                                              │
                                                      docker-build+scan
                                                              │
                                                      deploy to staging
                                                              │
Push to main    ──▶  (same) ──────────────────────▶ deploy to production
```

GitHub Environments used: `staging` and `production` (with required reviewers on production).

---

## Monitoring

- **Prometheus** metrics exposed at `/metrics` (scraped by Prometheus/Grafana)
- **Structured JSON logs** via structlog → stdout → log aggregator
- **Prediction audit log** persisted to PostgreSQL, queryable via `/transactions`
- **Drift alerts** logged by Airflow and triggerable as Slack notifications

---



