"""Tests for CSV formula safety, field order, and serialization consistency."""

from __future__ import annotations

import csv
import io
import json

import pandas as pd
import pytest

from bankruptcy_prediction.contracts import BatchSummary, RowStatus
from bankruptcy_prediction.export.csv_exporter import export_results_csv
from bankruptcy_prediction.scoring.result_models import (
    RESULT_COLUMNS,
    RiskClass,
    ScoringResult,
    ScoringResultRow,
)


def _result(record_id: str = "normal-id") -> ScoringResult:
    row = ScoringResultRow(
        record_id=record_id,
        model_version="model-v001",
        row_status=RowStatus.SUCCESS,
        confidence_score=0.76,
        batch_percentile=88.0,
        threshold_used=0.6,
        risk_class=RiskClass.HIGH_RISK,
        top_10_percent_flag=True,
        positive_factors=(),
        negative_factors=(),
    )
    return ScoringResult(BatchSummary(1, 1, 0, "model-v001", 0.6), (row,))


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@"])
def test_formula_like_record_ids_are_escaped_only_in_csv(prefix: str) -> None:
    result = _result(f"{prefix}SUM(1,2)")
    original_json = result.to_json()

    exported = export_results_csv(result)
    row = next(csv.DictReader(io.StringIO(exported.decode("utf-8"))))

    assert row["record_id"] == f"'{prefix}SUM(1,2)"
    assert result.rows[0].record_id == f"{prefix}SUM(1,2)"
    assert result.to_json() == original_json


def test_csv_field_order_and_json_equivalent_result_values() -> None:
    result = _result("一般公司-甲")

    exported = export_results_csv(result)
    frame = pd.read_csv(io.BytesIO(exported), dtype={"record_id": "string"})
    json_result = json.loads(result.to_json())

    assert tuple(frame.columns) == RESULT_COLUMNS
    assert frame.loc[0, "record_id"] == json_result["rows"][0]["record_id"]
    assert frame.loc[0, "model_version"] == json_result["rows"][0]["model_version"]
    assert frame.loc[0, "threshold_used"] == json_result["rows"][0]["threshold_used"]
    assert frame.loc[0, "row_status"] == json_result["rows"][0]["row_status"]
    assert frame.loc[0, "confidence_score"] == json_result["rows"][0]["confidence_score"]


def test_csv_quotes_newlines_quotes_and_unicode_correctly() -> None:
    result = _result('公司,"甲"\n第二行')

    data = export_results_csv(result).decode("utf-8")
    rows = list(csv.reader(io.StringIO(data)))

    assert rows[0] == list(RESULT_COLUMNS)
    assert rows[1][0] == '公司,"甲"\n第二行'
    assert data.startswith("record_id,")
