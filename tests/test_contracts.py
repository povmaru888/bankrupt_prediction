"""Serialization and invariant tests for shared data contracts."""

from __future__ import annotations

import json

import pytest

from bankruptcy_prediction.contracts import (
    BatchSummary,
    RowResult,
    RowStatus,
    WarningCode,
)
from bankruptcy_prediction.errors import ApiError


def _json_round_trip(value: object) -> dict[str, object]:
    return json.loads(json.dumps(value))


def test_success_row_contract_round_trips_with_warnings() -> None:
    row = RowResult(
        record_id="row_000001",
        row_status=RowStatus.SUCCESS,
        warning_codes=(WarningCode.IMPUTED_VALUES, WarningCode.OUT_OF_RANGE),
        confidence_score=0.73,
    )

    restored = RowResult.from_dict(_json_round_trip(row.to_dict()))
    assert restored == row
    assert restored.to_dict()["row_status"] == "success"
    assert restored.to_dict()["warning_codes"] == [
        "IMPUTED_VALUES",
        "out_of_range",
    ]


def test_failed_row_requires_error_fields() -> None:
    with pytest.raises(ValueError, match="failed rows require"):
        RowResult(record_id="row_000002", row_status=RowStatus.FAILED)

    failed = RowResult(
        record_id="row_000002",
        row_status=RowStatus.FAILED,
        error_code="INVALID_ROW",
        error_message="Row contains invalid values",
    )
    assert RowResult.from_dict(_json_round_trip(failed.to_dict())) == failed


def test_batch_summary_round_trips_and_checks_counts() -> None:
    summary = BatchSummary(3, 2, 1, "v0.1.0", 0.6)

    assert BatchSummary.from_dict(_json_round_trip(summary.to_dict())) == summary
    with pytest.raises(ValueError, match="must equal total_rows"):
        BatchSummary(3, 1, 1, "v0.1.0", 0.6)


def test_api_error_has_fixed_fields_and_round_trips() -> None:
    error = ApiError(
        "SCHEMA_INVALID", "CSV columns do not match", {"missing": ["x"]}, "Fix columns"
    )

    restored = ApiError.from_dict(_json_round_trip(error.to_dict()))
    assert restored == error
    assert set(restored.to_dict()) == {
        "error_code",
        "message",
        "details",
        "remediation",
    }


@pytest.mark.parametrize("field", ["error_code", "message", "remediation"])
def test_api_error_requires_all_human_and_machine_fields(field: str) -> None:
    payload = {
        "error_code": "INVALID",
        "message": "Invalid request",
        "details": {},
        "remediation": "Review the request",
    }
    payload[field] = " "

    with pytest.raises(ValueError, match=f"{field} must be a non-empty string"):
        ApiError(**payload)
