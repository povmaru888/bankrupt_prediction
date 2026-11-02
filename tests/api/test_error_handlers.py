from types import SimpleNamespace

from fastapi.testclient import TestClient

from bankruptcy_prediction.api.app import create_app
from bankruptcy_prediction.api.errors import PayloadTooLargeError


def test_request_id_is_returned_and_boundary_errors_have_fixed_envelope():
    app = create_app(SimpleNamespace())

    @app.get("/bad")
    def bad():
        raise PayloadTooLargeError("檔案過大。", details={"limit_mb": 10})

    response = TestClient(app).get("/bad", headers={"X-Request-ID": "req-123"})
    assert response.status_code == 413
    assert response.headers["X-Request-ID"] == "req-123"
    assert response.json() == {
        "error_code": "PAYLOAD_TOO_LARGE",
        "message": "檔案過大。",
        "details": {"limit_mb": 10},
        "remediation": "請縮小檔案或列數後重試。",
        "request_id": "req-123",
    }


def test_unexpected_error_is_sanitized():
    app = create_app(SimpleNamespace())

    @app.get("/crash")
    def crash():
        raise RuntimeError("D:\\secret\\uploaded.csv sensitive-feature")

    response = TestClient(app, raise_server_exceptions=False).get("/crash")
    assert response.status_code == 500
    payload = response.json()
    assert payload["error_code"] == "INTERNAL_ERROR"
    assert "secret" not in response.text
    assert "uploaded.csv" not in response.text


def test_app_ready_is_local_app_boundary():
    response = TestClient(create_app(SimpleNamespace())).get("/__app_ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
