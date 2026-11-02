from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from bankruptcy_prediction.api.app import create_app
from bankruptcy_prediction.contracts import BatchSummary, RowStatus
from bankruptcy_prediction.scoring.result_models import RiskClass, ScoringResult, ScoringResultRow


def make_result():
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


def settings(tmp_path, max_rows=10000, max_mb=10):
    return SimpleNamespace(
        service=SimpleNamespace(
            models_dir="models",
            temp_dir=str(tmp_path),
            max_rows=max_rows,
            max_upload_size_mb=max_mb,
        )
    )


def test_score_json_and_csv(monkeypatch, tmp_path):
    import bankruptcy_prediction.api.routes.score as route

    monkeypatch.setattr(route, "score_batch", lambda *args, **kwargs: make_result())
    app = create_app(settings(tmp_path), model_loader=lambda _: object())
    csv_data = pd.DataFrame({"f1": [1]}).to_csv(index=False).encode()
    client = TestClient(app)
    json_response = client.post(
        "/api/v1/score", files={"file": ("input.csv", csv_data, "text/csv")}
    )
    csv_response = client.post(
        "/api/v1/score?format=csv&threshold=0.7",
        files={"file": ("input.csv", csv_data, "text/csv")},
    )
    assert json_response.status_code == 200
    assert json_response.json()["rows"][0]["model_version"] == "v1"
    assert csv_response.status_code == 200
    assert "attachment" in csv_response.headers["content-disposition"]
    assert b"record_id" in csv_response.content


def test_score_rejects_row_limit_and_cleans_workspace(tmp_path):
    app = create_app(settings(tmp_path, max_rows=1), model_loader=lambda _: object())
    csv_data = b"f1\n1\n2\n"
    response = TestClient(app).post(
        "/api/v1/score", files={"file": ("input.csv", csv_data, "text/csv")}
    )
    assert response.status_code == 413
    assert response.json()["error_code"] == "PAYLOAD_TOO_LARGE"
    assert list(tmp_path.iterdir()) == []
