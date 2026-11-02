import json

import pytest

from bankruptcy_prediction.contracts import BatchSummary, RowStatus
from bankruptcy_prediction.scoring.response_formatter import (
    CsvResponse,
    JsonResponse,
    ResponseFormatError,
    format_score_response,
)
from bankruptcy_prediction.scoring.result_models import RiskClass, ScoringResult, ScoringResultRow


def make_result() -> ScoringResult:
    rows = (
        ScoringResultRow(
            record_id="a",
            model_version="v1",
            row_status=RowStatus.SUCCESS,
            confidence_score=0.6,
            batch_percentile=100,
            threshold_used=0.5,
            risk_class=RiskClass.HIGH_RISK,
            top_10_percent_flag=True,
            positive_factors=(),
            negative_factors=(),
        ),
        ScoringResultRow(
            record_id="b",
            model_version="v1",
            row_status=RowStatus.FAILED,
            threshold_used=0.5,
            error_code="BAD_ROW",
            error_message="bad",
        ),
    )
    return ScoringResult(BatchSummary(2, 1, 1, "v1", 0.5), rows)


def test_json_default_and_threshold_adjustment_do_not_mutate_source():
    result = make_result()
    response = format_score_response(result, "json", threshold=0.7)
    assert isinstance(response, JsonResponse)
    assert response.body["summary"]["threshold_used"] == 0.7
    assert response.body["rows"][0]["risk_class"] == "not_high_risk"
    assert response.body["rows"][0]["confidence_score"] == 0.6
    assert result.summary.threshold_used == 0.5
    json.loads(response.to_json())


def test_csv_and_json_have_same_threshold_and_rows():
    json_response = format_score_response(make_result(), "json", threshold=0.2)
    csv_response = format_score_response(make_result(), "csv", threshold=0.2)
    assert isinstance(csv_response, CsvResponse)
    assert b"threshold_used" in csv_response.body
    assert b",0.2," in csv_response.body
    assert json_response.body["rows"][1]["threshold_used"] == 0.2


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), "bad", True])
def test_invalid_threshold_is_file_level_error(value):
    with pytest.raises(ResponseFormatError) as error:
        format_score_response(make_result(), "json", threshold=value)
    assert error.value.error_code == "INVALID_THRESHOLD"
    assert set(error.value.to_dict()) == {"error_code", "message", "details", "remediation"}


def test_unknown_format_is_rejected():
    with pytest.raises(ResponseFormatError) as error:
        format_score_response(make_result(), "xml")
    assert error.value.error_code == "UNSUPPORTED_FORMAT"
