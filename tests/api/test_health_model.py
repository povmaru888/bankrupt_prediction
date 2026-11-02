from dataclasses import dataclass
from types import SimpleNamespace

from fastapi.testclient import TestClient

from bankruptcy_prediction.api.app import create_app
from bankruptcy_prediction.model_registry.registry import ActiveModelUnavailable


@dataclass(frozen=True)
class FakeSummary:
    def to_dict(self):
        return {
            "version": "v1",
            "model_family": "logistic_regression",
            "created_at": "2026-10-22T00:00:00Z",
            "test_only": True,
            "default_threshold": 0.5,
            "feature_count": 95,
            "average_precision": 0.4,
            "recall_at_top_10_percent": 0.6,
            "mvp_passed": False,
        }


@dataclass(frozen=True)
class FakeModel:
    version: str = "v1"
    summary: FakeSummary = FakeSummary()


def settings():
    return SimpleNamespace(service=SimpleNamespace(models_dir="models"))


def test_health_and_model_return_only_summary(monkeypatch):
    app = create_app(settings(), model_loader=lambda _: FakeModel())
    client = TestClient(app)
    health = client.get("/health")
    model = client.get("/api/v1/model")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "active_model_loaded": True, "model_version": "v1"}
    assert model.status_code == 200
    assert model.json()["feature_count"] == 95
    assert "models" not in model.text


def test_health_stays_available_without_model_and_model_endpoint_is_503():
    def unavailable(_):
        raise ActiveModelUnavailable("no_active_model", "local path must not leak")

    app = create_app(settings(), model_loader=unavailable)
    client = TestClient(app, raise_server_exceptions=False)
    health = client.get("/health")
    model = client.get("/api/v1/model")
    assert health.status_code == 200
    assert health.json() == {
        "status": "degraded",
        "active_model_loaded": False,
        "reason_code": "no_active_model",
    }
    assert model.status_code == 503
    assert model.json()["error_code"] == "no_active_model"
    assert "local path" not in model.text
