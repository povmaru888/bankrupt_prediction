"""FastAPI workflow tests using an isolated temporary workspace."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from bankruptcy_prediction.api.app import create_app
from bankruptcy_prediction.contracts import BatchSummary, RowStatus
from bankruptcy_prediction.model_registry.registry import ActiveModelUnavailable
from bankruptcy_prediction.scoring.result_models import RiskClass, ScoringResult, ScoringResultRow


def settings(tmp_path, max_rows=10000):
    return SimpleNamespace(
        service=SimpleNamespace(
            models_dir="models",
            temp_dir=str(tmp_path),
            max_rows=max_rows,
            max_upload_size_mb=10,
        )
    )


def result():
    row = ScoringResultRow(
        record_id="row_000001",
        model_version="v1",
        row_status=RowStatus.SUCCESS,
        confidence_score=0.8,
        batch_percentile=100,
        threshold_used=0.5,
        risk_class=RiskClass.HIGH_RISK,
        top_10_percent_flag=True,
        positive_factors=(),
        negative_factors=(),
    )
    return ScoringResult(BatchSummary(1, 1, 0, "v1", 0.5), (row,))


def test_api_workflow_json_csv_and_request_id(monkeypatch, tmp_path):
    import bankruptcy_prediction.api.routes.score as score_route

    monkeypatch.setattr(score_route, "score_batch", lambda *args, **kwargs: result())
    app = create_app(
        settings(tmp_path),
        model_loader=lambda _: SimpleNamespace(
            version="v1", summary=SimpleNamespace(to_dict=lambda: {"version": "v1"})
        ),
    )
    client = TestClient(app)
    data = b"f1\n1\n"
    json_response = client.post(
        "/api/v1/score",
        files={"file": ("input.csv", data, "text/csv")},
        headers={"X-Request-ID": "workflow-1"},
    )
    csv_response = client.post(
        "/api/v1/score?format=csv", files={"file": ("input.csv", data, "text/csv")}
    )
    assert json_response.status_code == 200
    assert json_response.headers["x-request-id"] == "workflow-1"
    assert json_response.json()["summary"]["model_version"] == "v1"
    assert csv_response.status_code == 200
    assert b"record_id" in csv_response.content
    assert list(tmp_path.iterdir()) == []


def test_api_workflow_covers_422_413_and_503(monkeypatch, tmp_path):
    import bankruptcy_prediction.api.routes.score as score_route

    monkeypatch.setattr(score_route, "score_batch", lambda *args, **kwargs: result())
    app = create_app(settings(tmp_path, max_rows=1), model_loader=lambda _: object())
    client = TestClient(app, raise_server_exceptions=False)
    assert (
        client.post(
            "/api/v1/score?format=xml", files={"file": ("x.csv", b"f1\n1\n", "text/csv")}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/score", files={"file": ("x.csv", b"f1\n1\n2\n", "text/csv")}
        ).status_code
        == 413
    )

    def unavailable(_):
        raise ActiveModelUnavailable("no_active_model", "hidden")

    unavailable_app = create_app(settings(tmp_path), model_loader=unavailable)
    response = TestClient(unavailable_app, raise_server_exceptions=False).post(
        "/api/v1/score", files={"file": ("x.csv", b"f1\n1\n", "text/csv")}
    )
    assert response.status_code == 503
    assert list(tmp_path.iterdir()) == []
