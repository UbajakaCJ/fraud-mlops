# 🛡️ Fraud Detection MLOps Platform

**Bluechip Technologies — AI Services Department**

A production-grade, end-to-end MLOps platform for real-time credit card fraud detection. Built on FastAPI, MongoDB, Apache Airflow, and MLflow — fully containerised with Docker Compose and battle-tested with concurrent load and real-time simulation scripts.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     nginx  (port 80)                         │
│                  Rate limit: 300 req/s                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   FastAPI  (port 8000)  │
          │   4 uvicorn workers     │
          │   JWT authentication    │
          └──┬──────────┬──────────┘
             │          │
    ┌─────────▼──┐  ┌───▼──────────┐
    │  MongoDB   │  │    Redis      │
    │  frauddb   │  │    cache      │
    │ audit trail│  │  (TTL 300s)   │
    └────────────┘  └──────────────┘
             │
    ┌─────────▼───────────┐   ┌──────────────────┐
    │  MLflow (port 5000) │   │  Airflow          │
    │  Experiment tracking│   │  (port 8081)      │
    │  Model registry     │   │  Retraining DAGs  │
    └─────────────────────┘   │  Drift monitoring │
                              └──────────────────┘
```

---

## ✨ Features

- **Real-time prediction** — single transaction scoring with confidence intervals and sub-20ms API latency
- **Batch prediction** — bulk scoring up to 10,000 transactions at ~10,000 TPS
- **MongoDB audit trail** — every prediction logged to `frauddb.transaction_logs` with full reproducibility
- **Model management** — register, promote, and roll back models via MLflow champion/challenger pattern
- **Experiment tracking** — full MLflow lineage across all retraining runs
- **Drift detection** — feature drift monitoring with configurable alert thresholds
- **Automated retraining** — Airflow DAGs for scheduled retraining and drift response
- **JWT authentication** — Bearer token security across all prediction endpoints
- **Swagger UI** — interactive API docs at `/docs` with Authorize button
- **Prometheus metrics** — instrumented via `prometheus-fastapi-instrumentator`

---

## 🚀 Quickstart

### Prerequisites

- Docker Desktop
- Docker Compose v2+
- Git

### 1. Clone

```bash
git clone https://github.com/UbajakaCJ/fraud-mlops.git
cd fraud-mlops
```

### 2. Configure

```bash
cp .env.example .env
```

The defaults work out of the box for local Docker:

```env
MONGODB_URI=mongodb://mongodb:27017
MONGODB_DB_NAME=frauddb
MLFLOW_TRACKING_URI=http://mlflow:5000
REDIS_URL=redis://redis:6379/0
SECRET_KEY=your-32-char-secret-key-here
```

### 3. Build and start

```bash
docker compose up --build
```

First run takes 5–10 minutes. Once healthy:

| Service | URL |
|---|---|
| **API Swagger UI** | http://localhost/docs |
| **Airflow** | http://localhost:8081 |
| **MLflow** | http://localhost:5000 |

### 4. Create Airflow admin (first time only)

```bash
docker compose exec airflow-webserver airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com
```

### 5. Train the model

```bash
python -c "
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X_train = joblib.load('app/models/splits/X_train_eng.pkl')
y_train = joblib.load('app/models/splits/y_train.pkl')

model = Pipeline([('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
model.fit(X_train, y_train)
joblib.dump(model, 'model/fraud_model_v1.pkl')
print('Model saved')
"
```

Then restart the API:

```bash
docker compose restart api
```

---

## 📡 API Reference

### Authenticate

```http
POST /auth/token
Content-Type: application/json

{"username": "admin", "password": "admin"}
```

Returns a JWT bearer token. Pass it as `Authorization: Bearer <token>` on all subsequent requests.

### Single prediction

```http
POST /predict/
Authorization: Bearer <token>
Content-Type: application/json

{
  "transaction_id": "TXN-2024-000001",
  "transaction": {
    "V1": -1.36, "V2": -0.07, "V3": 2.54, ..., "V28": -0.02,
    "Amount": 149.62,
    "Time": 0.0
  }
}
```

**Response:**

```json
{
  "transaction_id": "TXN-2024-000001",
  "is_fraud": false,
  "fraud_probability": 0.156,
  "risk_level": "LOW",
  "confidence": "medium",
  "threshold_used": 0.33,
  "model_name": "fraud_model_v1",
  "model_version": "local",
  "latency_ms": 7.37,
  "timestamp": "2026-04-29T07:00:46.634Z"
}
```

Risk levels: `LOW` · `MEDIUM` · `HIGH` · `CRITICAL`

### Batch prediction

```http
POST /predict/batch
Authorization: Bearer <token>
Content-Type: application/json

{"transactions": [{...}, {...}, ...]}
```

Accepts up to 10,000 transactions per request.

### Audit log

```http
GET /transactions/?page=1&size=20&fraud_only=false
Authorization: Bearer <token>
```

Returns a paginated audit trail of all predictions stored in MongoDB.

### Health check

```http
GET /health
```

Returns status of all dependencies: API, model, MongoDB, Redis, MLflow.

---

## 🧠 Model

The fraud detection model is a scikit-learn `Pipeline` trained on the [IEEE-CIS Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (284,807 transactions, 0.172% fraud rate).

**Feature engineering applied at inference:**

| Feature | Description |
|---|---|
| `V1`–`V28` | PCA-transformed card features |
| `log(Amount + 1)` | Log-normalised transaction amount |
| `time_hour` | Hour of day from `Time` field |
| `time_sin` / `time_cos` | Cyclical encoding of hour |

**Classification threshold:** `0.33` (tunable via `PREDICTION_THRESHOLD` env var)

---

## 🧪 Testing & Load Simulation

### Real-time fraud simulator

Streams transactions one by one, mimicking a live payment feed:

```powershell
# Default: 5 txns/sec, 50 transactions
.\fraud_simulator.ps1

# Full 1,000 transaction run, fraud alerts only
.\fraud_simulator.ps1 -TransactionsPerSec 50 -MaxTransactions 1000 -AlertsOnly

# Use a larger fixture
.\fraud_simulator.ps1 -FixtureFile ".\tests\fixtures\batch_5000.json" -TransactionsPerSec 20 -MaxTransactions 200
```

### Concurrent load test

```powershell
.\loadtest.ps1 -Workers 25 -Duration 30
```

### Batch throughput test

```powershell
.\loadtest_batch.ps1
```

**Benchmark results (Quadro T2000, 4 uvicorn workers):**

| Test | Result |
|---|---|
| Concurrent success rate | **99.98%** (5,824 / 5,825) |
| Concurrent TPS | **~93 req/sec** |
| Batch TPS (1k) | **7,604 txns/sec** |
| Batch TPS (10k) | **9,816 txns/sec** |
| P50 latency | **115ms** |
| P95 latency | **266ms** |
| Fraud detection rate | **~17.7%** on IEEE-CIS dataset |
| Scaling efficiency | **77.5%** (sub-linear — vectorised inference confirmed) |

---

## 📁 Project Structure

```
fraud-mlops/
├── app/
│   ├── core/
│   │   ├── database.py              # MongoDB Motor async client
│   │   ├── config.py                # Pydantic settings
│   │   └── model_registry.py        # MLflow model loader
│   ├── models/
│   │   └── transaction.py           # MongoDB document model
│   ├── routers/
│   │   ├── predict.py               # /predict/ endpoints
│   │   ├── transactions.py          # /transactions/ audit log
│   │   ├── health.py                # /health dependency checks
│   │   ├── models.py                # Model registry management
│   │   ├── experiments.py           # MLflow experiment tracking
│   │   └── auth.py                  # JWT authentication
│   ├── schemas/
│   │   └── schemas.py               # Pydantic request/response models
│   ├── services/
│   │   ├── audit_logger.py          # MongoDB async audit writer
│   │   └── feature_engineering.py   # Inference-time feature transforms
│   └── main.py                      # FastAPI app + lifespan hooks
├── airflow/
│   └── dags/
│       ├── fraud_retrain_pipeline.py
│       └── fraud_drift_monitor.py
├── model/                           # Trained .pkl files (gitignored)
├── tests/
│   └── fixtures/
│       ├── batch_1000.json
│       ├── batch_5000.json
│       └── batch_10000.json
├── nginx/
│   └── nginx.conf
├── fraud_simulator.ps1              # Real-time detection simulator
├── loadtest.ps1                     # Concurrent load test
├── loadtest_batch.ps1               # Batch throughput test
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 🐳 Docker Services

| Service | Image | Port |
|---|---|---|
| `api` | `fraud-mlops-api` | 8000 |
| `nginx` | `nginx:1.25-alpine` | 80 |
| `mongodb` | `mongo:7.0` | 27017 |
| `redis` | `redis:7-alpine` | 6379 |
| `mlflow` | `fraud-mlops-mlflow` | 5000 |
| `airflow-webserver` | `fraud-mlops-airflow` | 8081 |
| `airflow-scheduler` | `fraud-mlops-airflow` | — |
| `postgres` | `postgres:15-alpine` | 5432 (Airflow metadata) |

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MONGODB_URI` | `mongodb://mongodb:27017` | MongoDB connection string |
| `MONGODB_DB_NAME` | `frauddb` | Database name |
| `SECRET_KEY` | — | JWT signing key (32+ chars) |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `REDIS_URL` | `redis://redis:6379/0` | Redis cache URL |
| `MODEL_PATH` | `/app/model` | Directory containing `.pkl` model files |
| `PREDICTION_THRESHOLD` | `0.33` | Fraud classification threshold |
| `BATCH_LIMIT` | `10000` | Maximum transactions per batch request |
| `ENVIRONMENT` | `production` | Runtime environment label |
| `LOG_LEVEL` | `info` | Logging verbosity |

---

## 👤 Author

**Chijioke Ubajaka** — Senior Data Scientist / ML Engineer  
Bluechip Technologies — AI Services Department, Lagos  
📧 cubajaka@bluechiptech.biz  
🔗 [GitHub: UbajakaCJ](https://github.com/UbajakaCJ)

---

## 📄 License

Proprietary — Bluechip Technologies. All rights reserved.
