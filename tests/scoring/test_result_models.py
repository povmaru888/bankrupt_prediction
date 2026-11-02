"""Tests for result model JSON and DataFrame schemas."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bankruptcy_prediction.contracts import BatchSummary, RowStatus, WarningCode
from bankruptcy_prediction.scoring.result_models import (
    RESULT_COLUMNS,
    FactorContribution,
    RiskClass,
    ScoringResult,
    ScoringResultRow,
    results_from_dataframe,
)


def _success_row() -> ScoringResultRow:
    return ScoringResultRow(
        record_id="row_000001",
        model_version="model-v001",
        row_status=RowStatus.SUCCESS,
        confidence_score=0.84,
        batch_percentile=95.0,
        threshold_used=0.62,
        risk_class=RiskClass.HIGH_RISK,
        top_10_percent_flag=True,
        warning_codes=(WarningCode.IMPUTED_VALUES,),
        positive_factors=(FactorContribution("Debt ratio %", 0.15),),
        negative_factors=(FactorContribution("Cash flow rate", -0.08),),
    )


def test_success_and_failure_rows_round_trip_json_and_fixed_dataframe_schema() -> None:
    success = _success_row()
    failure = ScoringResultRow(
        record_id="row_000002",
        model_version="model-v001",
        row_status=RowStatus.FAILED,
        threshold_used=0.62,
        error_code="NON_NUMERIC_VALUE",
        error_message="One or more features are not numeric",
    )
    result = ScoringResult(BatchSummary(2, 1, 1, "model-v001", 0.62), (success, failure))

    restored_json = ScoringResult.from_dict(json.loads(result.to_json()))
    assert restored_json == result

    frame = result.to_dataframe()
    assert tuple(frame.columns) == RESULT_COLUMNS
    assert frame.loc[0, "row_status"] == "success"
    assert frame.loc[1, "row_status"] == "failed"
    assert pd.isna(frame.loc[1, "confidence_score"])
    assert pd.isna(frame.loc[1, "batch_percentile"])
    assert pd.isna(frame.loc[1, "risk_class"])
    assert pd.isna(frame.loc[1, "positive_factors"])
    assert results_from_dataframe(frame, result.summary) == result


def test_success_rows_require_all_scoring_fields_and_bounded_explanations() -> None:
    with pytest.raises(ValueError, match="successful rows require confidence_score"):
        ScoringResultRow(
            record_id="row-1",
            model_version="v1",
            row_status=RowStatus.SUCCESS,
            threshold_used=0.5,
        )

    with pytest.raises(ValueError, match="at most three factors"):
        ScoringResultRow(
            record_id="row-1",
            model_version="v1",
            row_status=RowStatus.SUCCESS,
            confidence_score=0.5,
            batch_percentile=50,
            threshold_used=0.5,
            risk_class=RiskClass.NOT_HIGH_RISK,
            top_10_percent_flag=False,
            positive_factors=tuple(FactorContribution(f"f{i}", 0.1) for i in range(4)),
            negative_factors=(),
        )


def test_failed_rows_cannot_carry_predictions_or_explanations() -> None:
    with pytest.raises(ValueError, match="cannot contain scoring or explanation"):
        ScoringResultRow(
            record_id="row-1",
            model_version="v1",
            row_status=RowStatus.FAILED,
            threshold_used=0.5,
            confidence_score=0.5,
            error_code="INVALID",
            error_message="Invalid row",
        )


def test_dataframe_projection_rejects_changed_column_order() -> None:
    frame = pd.DataFrame(columns=list(reversed(RESULT_COLUMNS)))
    with pytest.raises(ValueError, match="fixed result schema"):
        results_from_dataframe(frame, BatchSummary(0, 0, 0, "v1", 0.5))
