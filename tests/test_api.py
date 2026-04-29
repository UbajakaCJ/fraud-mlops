"""
Test Suite — Fraud Detection MLOps API
Run: pytest tests/test_api.py -v --tb=short
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from httpx import AsyncClient, ASGITransport

# ─── Shared test data ─────────────────────────────────────────────────────────

VALID_TRANSACTION = {
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624,  "V7": 0.2396, "V8": 0.0987,
    "V9": 0.3638,  "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
    "V17": 0.2080,  "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
    "V25": 0.1285,  "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
    "Amount": 149.62, "Time": 0.0,
}

# ─── Mock factory ─────────────────────────────────────────────────────────────

def make_mock_registry(probs=None):
    if probs is None:
        probs = np.array([0.85])
    preds = (probs >= 0.33).astype(int)
    m = MagicMock()
    m.is_ready.return_value = True
    m.champion_name    = "test-model"
    m.champion_version = "1"
    m.champion_run_id  = "abc123"
    m.info.return_value = {
        "name": "test-model", "version": "1",
        "run_id": "abc123", "threshold": 0.33, "ready": True,
    }
    m.predict.return_value = (preds, probs)
    return m


# ─── Shared app context manager ───────────────────────────────────────────────

@asynccontextmanager
async def api_client(registry=None):
    """Yield an AsyncClient with all external I/O patched out."""
    from app.main import app
    from app.schemas.schemas import DependencyHealth, HealthStatus
    if registry is None:
        registry = make_mock_registry()

    dep_ok = DependencyHealth(name="test", status=HealthStatus.OK, latency_ms=1.0, detail=None)

    # Full async session mock so predict routes don't touch Postgres
    mock_session = MagicMock()
    mock_session.commit   = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close    = AsyncMock()
    mock_session.add      = MagicMock()

    async def _fake_get_db():
        yield mock_session

    with (
        patch("app.core.database.engine"),
        patch("app.core.database.get_db",   _fake_get_db),
        patch("app.routers.predict.get_db",  _fake_get_db),
        patch("app.routers.transactions.get_db", _fake_get_db),
        patch("app.core.database.AsyncSessionLocal", return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )),
        patch("app.core.model_registry.model_registry", registry),
        patch("app.routers.predict.model_registry",     registry),
        patch("app.routers.models.model_registry",      registry),
        patch("app.routers.health.model_registry",      registry),
        patch("app.services.cache.get_cached_prediction",  new=AsyncMock(return_value=None)),
        patch("app.services.cache.set_cached_prediction",  new=AsyncMock()),
        patch("app.services.audit_logger.log_prediction",  new=AsyncMock()),
        patch("app.routers.health._check_postgres",  new=AsyncMock(return_value=dep_ok)),
        patch("app.routers.health._check_redis",     new=AsyncMock(return_value=dep_ok)),
        patch("app.routers.health._check_mlflow",    new=AsyncMock(return_value=dep_ok)),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Pure functions, no I/O
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    def test_output_shape_is_1x32(self):
        from app.services.feature_engineering import engineer_features
        arr = engineer_features(VALID_TRANSACTION)
        assert arr.shape == (1, 32)  # 28 V + log_amount + time_hour + sin + cos

    def test_log_amount_correct(self):
        from app.services.feature_engineering import engineer_features
        txn = {**VALID_TRANSACTION, "Amount": np.e - 1}
        arr = engineer_features(txn)
        assert abs(arr[0, 28] - 1.0) < 1e-5

    def test_zero_amount_gives_log1p_zero(self):
        from app.services.feature_engineering import engineer_features
        arr = engineer_features({**VALID_TRANSACTION, "Amount": 0.0})
        assert arr[0, 28] == pytest.approx(0.0)

    def test_cyclical_time_unit_circle(self):
        from app.services.feature_engineering import engineer_features
        for t in [0.0, 3600.0, 43200.0, 86399.0]:
            arr = engineer_features({**VALID_TRANSACTION, "Time": t})
            sin_v, cos_v = arr[0, 30], arr[0, 31]
            assert abs(sin_v**2 + cos_v**2 - 1.0) < 1e-5

    def test_output_dtype_float32(self):
        from app.services.feature_engineering import engineer_features
        arr = engineer_features(VALID_TRANSACTION)
        assert arr.dtype == np.float32

    def test_sin_cos_in_unit_range(self):
        from app.services.feature_engineering import engineer_features
        arr = engineer_features(VALID_TRANSACTION)
        assert -1.0 <= arr[0, 30] <= 1.0
        assert -1.0 <= arr[0, 31] <= 1.0


class TestRiskLevel:
    @pytest.mark.parametrize("prob,expected", [
        (0.00, "LOW"), (0.10, "LOW"), (0.24, "LOW"),
        (0.25, "MEDIUM"), (0.40, "MEDIUM"), (0.49, "MEDIUM"),
        (0.50, "HIGH"), (0.65, "HIGH"), (0.74, "HIGH"),
        (0.75, "CRITICAL"), (0.99, "CRITICAL"), (1.00, "CRITICAL"),
    ])
    def test_risk_boundaries(self, prob, expected):
        from app.routers.predict import _risk_level
        from app.schemas.schemas import RiskLevel
        assert _risk_level(prob) == RiskLevel(expected)

    @pytest.mark.parametrize("prob,expected_conf", [
        (0.01, "high"),       # dist from threshold = |0.01-0.33| = 0.32, in (0.25, 0.4) → "high"
        (0.99, "very_high"),  # dist = 0.66 > 0.4 → very_high
        (0.30, "low"),        # dist = 0.03 < 0.1 → low
        (0.36, "low"),        # dist = 0.03 < 0.1 → low
    ])
    def test_confidence_classification(self, prob, expected_conf):
        from app.routers.predict import _confidence
        assert _confidence(prob) == expected_conf


class TestSchemaValidation:
    def test_valid_transaction_parses(self):
        from app.schemas.schemas import TransactionFeatures
        txn = TransactionFeatures(**VALID_TRANSACTION)
        assert txn.Amount == 149.62

    def test_negative_amount_raises_validation_error(self):
        from app.schemas.schemas import TransactionFeatures
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TransactionFeatures(**{**VALID_TRANSACTION, "Amount": -1.0})

    def test_missing_feature_raises_validation_error(self):
        from app.schemas.schemas import TransactionFeatures
        from pydantic import ValidationError
        bad = {k: v for k, v in VALID_TRANSACTION.items() if k != "V14"}
        with pytest.raises(ValidationError):
            TransactionFeatures(**bad)

    def test_batch_request_empty_raises(self):
        from app.schemas.schemas import BatchPredictRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BatchPredictRequest(transactions=[])

    def test_batch_request_single_valid(self):
        from app.schemas.schemas import BatchPredictRequest, TransactionFeatures
        req = BatchPredictRequest(transactions=[TransactionFeatures(**VALID_TRANSACTION)])
        assert len(req.transactions) == 1

    def test_model_promote_short_justification_raises(self):
        from app.schemas.schemas import ModelPromoteRequest, ModelStage
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ModelPromoteRequest(
                model_name="m", version="1",
                target_stage=ModelStage.PRODUCTION,
                justification="short",
            )

    def test_model_promote_valid(self):
        from app.schemas.schemas import ModelPromoteRequest, ModelStage
        req = ModelPromoteRequest(
            model_name="fraud-detector-champion",
            version="7",
            target_stage=ModelStage.PRODUCTION,
            justification="AUPRC improved from 0.87 to 0.91 on holdout set.",
        )
        assert req.version == "7"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — FastAPI routes via async HTTP client
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthRoutes:
    @pytest.mark.asyncio
    async def test_health_200_with_expected_fields(self):
        async with api_client() as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        for field in ["status", "version", "environment", "model", "dependencies", "timestamp"]:
            assert field in data, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_health_model_ready_true(self):
        async with api_client() as ac:
            resp = await ac.get("/health")
        assert resp.json()["model"]["ready"] is True

    @pytest.mark.asyncio
    async def test_root_returns_service_name(self):
        async with api_client() as ac:
            resp = await ac.get("/")
        assert resp.status_code == 200
        assert "Fraud Detection" in resp.json()["service"]


class TestAuthRoutes:
    @pytest.mark.asyncio
    async def test_admin_login_returns_token(self):
        async with api_client() as ac:
            resp = await ac.post("/auth/token", json={"username": "admin", "password": "admin"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert isinstance(data["expires_in"], int) and data["expires_in"] > 0

    @pytest.mark.asyncio
    async def test_viewer_login_succeeds(self):
        async with api_client() as ac:
            resp = await ac.post("/auth/token", json={"username": "viewer", "password": "viewer"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_wrong_password_returns_401(self):
        async with api_client() as ac:
            resp = await ac.post("/auth/token", json={"username": "admin", "password": "notthepassword"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_user_returns_401(self):
        async with api_client() as ac:
            resp = await ac.post("/auth/token", json={"username": "ghost", "password": "pass"})
        assert resp.status_code == 401


class TestPredictRoutes:
    @pytest.mark.asyncio
    async def test_single_prediction_full_response_shape(self):
        async with api_client() as ac:
            resp = await ac.post("/predict/", json={"transaction": VALID_TRANSACTION})
        assert resp.status_code == 200
        data = resp.json()
        expected_keys = [
            "transaction_id", "is_fraud", "fraud_probability",
            "risk_level", "confidence", "threshold_used",
            "model_name", "model_version", "latency_ms", "timestamp",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key in response: {key}"

    @pytest.mark.asyncio
    async def test_high_fraud_prob_flagged_as_fraud(self):
        async with api_client(make_mock_registry(probs=np.array([0.92]))) as ac:
            resp = await ac.post("/predict/", json={"transaction": VALID_TRANSACTION})
        data = resp.json()
        assert data["is_fraud"] is True
        assert data["risk_level"] == "CRITICAL"

    @pytest.mark.asyncio
    async def test_low_fraud_prob_not_flagged(self):
        async with api_client(make_mock_registry(probs=np.array([0.05]))) as ac:
            resp = await ac.post("/predict/", json={"transaction": VALID_TRANSACTION})
        data = resp.json()
        assert data["is_fraud"] is False
        assert data["risk_level"] == "LOW"

    @pytest.mark.asyncio
    async def test_custom_txn_id_echoed_back(self):
        async with api_client() as ac:
            resp = await ac.post("/predict/", json={
                "transaction_id": "MYID-XYZ-999",
                "transaction": VALID_TRANSACTION,
            })
        assert resp.json()["transaction_id"] == "MYID-XYZ-999"

    @pytest.mark.asyncio
    async def test_auto_txn_id_starts_with_TXN(self):
        async with api_client() as ac:
            resp = await ac.post("/predict/", json={"transaction": VALID_TRANSACTION})
        assert resp.json()["transaction_id"].startswith("TXN-")

    @pytest.mark.asyncio
    async def test_missing_feature_returns_422(self):
        bad = {k: v for k, v in VALID_TRANSACTION.items() if k != "V7"}
        async with api_client() as ac:
            resp = await ac.post("/predict/", json={"transaction": bad})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_negative_amount_returns_422(self):
        async with api_client() as ac:
            resp = await ac.post("/predict/", json={
                "transaction": {**VALID_TRANSACTION, "Amount": -100.0}
            })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_model_not_ready_returns_503(self):
        reg = make_mock_registry()
        reg.is_ready.return_value = False
        async with api_client(reg) as ac:
            resp = await ac.post("/predict/", json={"transaction": VALID_TRANSACTION})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_threshold_used_matches_config(self):
        async with api_client() as ac:
            resp = await ac.post("/predict/", json={"transaction": VALID_TRANSACTION})
        assert resp.json()["threshold_used"] == 0.33


class TestBatchPredictRoutes:
    @pytest.mark.asyncio
    async def test_batch_5_correct_aggregates(self):
        probs = np.array([0.85, 0.10, 0.92, 0.05, 0.78])
        async with api_client(make_mock_registry(probs=probs)) as ac:
            resp = await ac.post("/predict/batch", json={
                "transactions": [VALID_TRANSACTION] * 5
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert data["fraud_count"] == 3   # 0.85, 0.92, 0.78 ≥ 0.33
        assert data["fraud_rate_pct"] == pytest.approx(60.0, abs=0.01)
        assert len(data["results"]) == 5

    @pytest.mark.asyncio
    async def test_batch_all_legit(self):
        probs = np.array([0.01, 0.02, 0.03])
        async with api_client(make_mock_registry(probs=probs)) as ac:
            resp = await ac.post("/predict/batch", json={
                "transactions": [VALID_TRANSACTION] * 3
            })
        data = resp.json()
        assert data["fraud_count"] == 0
        assert data["fraud_rate_pct"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_batch_unique_txn_ids(self):
        probs = np.array([0.5, 0.6, 0.7])
        async with api_client(make_mock_registry(probs=probs)) as ac:
            resp = await ac.post("/predict/batch", json={
                "transactions": [VALID_TRANSACTION] * 3
            })
        ids = [r["transaction_id"] for r in resp.json()["results"]]
        assert len(set(ids)) == 3

    @pytest.mark.asyncio
    async def test_batch_empty_returns_422(self):
        async with api_client() as ac:
            resp = await ac.post("/predict/batch", json={"transactions": []})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_model_not_ready_503(self):
        reg = make_mock_registry()
        reg.is_ready.return_value = False
        async with api_client(reg) as ac:
            resp = await ac.post("/predict/batch", json={
                "transactions": [VALID_TRANSACTION] * 2
            })
        assert resp.status_code == 503


class TestDriftRoute:
    @pytest.mark.asyncio
    async def test_drift_report_required_keys(self):
        async with api_client() as ac:
            resp = await ac.get("/predict/drift")
        assert resp.status_code == 200
        data = resp.json()
        for key in ["drift_scores", "features_drifted", "alert_triggered", "recommendation"]:
            assert key in data

    @pytest.mark.asyncio
    async def test_drift_scores_are_numeric(self):
        async with api_client() as ac:
            resp = await ac.get("/predict/drift")
        for feat, score in resp.json()["drift_scores"].items():
            assert isinstance(score, (int, float)), f"{feat}: {score} not numeric"

    @pytest.mark.asyncio
    async def test_features_drifted_is_list(self):
        async with api_client() as ac:
            resp = await ac.get("/predict/drift")
        assert isinstance(resp.json()["features_drifted"], list)


class TestOpenAPISchema:
    @pytest.mark.asyncio
    async def test_openapi_json_valid(self):
        async with api_client() as ac:
            resp = await ac.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "🛡️ Fraud Detection MLOps API"
        assert "3." in schema["openapi"]   # OpenAPI 3.x

    @pytest.mark.asyncio
    async def test_all_critical_paths_documented(self):
        async with api_client() as ac:
            schema = (await ac.get("/openapi.json")).json()
        paths = schema["paths"]
        for required in [
            "/predict/", "/predict/batch", "/predict/drift",
            "/health", "/auth/token",
            "/models/champion", "/models/promote", "/models/reload",
            "/experiments/",
        ]:
            assert required in paths, f"Path not documented: {required}"

    @pytest.mark.asyncio
    async def test_swagger_ui_renders(self):
        async with api_client() as ac:
            resp = await ac.get("/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_redoc_renders(self):
        async with api_client() as ac:
            resp = await ac.get("/redoc")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_api_tags_present(self):
        async with api_client() as ac:
            schema = (await ac.get("/openapi.json")).json()
        tag_names = {t["name"] for t in schema.get("tags", [])}
        for required_tag in ["predict", "models", "experiments", "health", "auth"]:
            assert required_tag in tag_names, f"Missing tag: {required_tag}"


# ═══════════════════════════════════════════════════════════════════════════════
# DRIFT DAG UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPSIUtility:
    """Tests for PSI utility — imports directly from app.services.psi (no Airflow dependency)."""

    def test_identical_distributions_psi_near_zero(self):
        from app.services.psi import compute_psi
        x = np.random.default_rng(42).standard_normal(2000)
        assert compute_psi(x, x) < 0.02

    def test_same_distribution_different_samples(self):
        from app.services.psi import compute_psi
        rng = np.random.default_rng(0)
        x = rng.standard_normal(2000)
        y = rng.standard_normal(2000)
        assert compute_psi(x, y) < 0.10

    def test_3sigma_shift_exceeds_warning_threshold(self):
        from app.services.psi import compute_psi
        rng = np.random.default_rng(1)
        x = rng.standard_normal(2000)
        y = rng.standard_normal(2000) + 3.0
        assert compute_psi(x, y) > 0.10

    def test_5sigma_shift_exceeds_critical_threshold(self):
        from app.services.psi import compute_psi
        rng = np.random.default_rng(2)
        x = rng.standard_normal(2000)
        y = rng.standard_normal(2000) + 5.0
        assert compute_psi(x, y) > 0.20

    def test_psi_always_non_negative(self):
        from app.services.psi import compute_psi
        rng = np.random.default_rng(3)
        for _ in range(15):
            x = rng.standard_normal(500)
            y = rng.standard_normal(500) + rng.uniform(-3, 3)
            assert compute_psi(x, y) >= 0.0

    def test_psi_returns_float(self):
        from app.services.psi import compute_psi
        rng = np.random.default_rng(99)
        x = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        result = compute_psi(x, y)
        assert isinstance(result, float)
