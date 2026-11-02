from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bankruptcy_prediction.contracts import RowStatus, WarningCode
from bankruptcy_prediction.explanation.shap_service import (
    ExplanationBatch,
    ExplanationFactor,
    ExplanationRow,
)
from bankruptcy_prediction.scoring import service
from bankruptcy_prediction.scoring.service import score_batch

FEATURES = ("f1", "f2", "f3")


class FakeModel:
    def predict_proba(self, frame):
        values = pd.to_numeric(frame["f1"], errors="coerce").fillna(0).to_numpy(float)
        scores = np.clip(values / 10, 0.0, 1.0)
        return np.column_stack([1 - scores, scores])


def make_active(range_monitor=None):
    active = SimpleNamespace(
        version="v-test",
        model=FakeModel(),
        feature_names=FEATURES,
        feature_schema={"schema_version": 1, "feature_names": FEATURES},
        default_threshold=0.5,
        model_family="logistic_regression",
    )
    if range_monitor is not None:
        active.range_monitor = range_monitor
    return active


def fake_explanations(model_family, model, frame):
    del model_family, model
    rows = tuple(
        ExplanationRow(
            positive_factors=(ExplanationFactor("f1", float(value), 0.2),),
            negative_factors=(ExplanationFactor("f2", 1.0, -0.1),),
        )
        for value in frame["f1"].to_numpy(float)
    )
    return ExplanationBatch(rows=rows, feature_names=FEATURES, model_family="logistic_regression")


@pytest.fixture(autouse=True)
def patch_explain(monkeypatch):
    monkeypatch.setattr(service, "explain", fake_explanations)


def test_valid_batch_uses_average_rank_percentile_and_stable_top_ten():
    frame = pd.DataFrame({"f1": [2, 2, 8, 1], "f2": [1, 1, 1, 1], "f3": [0, 0, 0, 0]})
    result = score_batch(make_active(), frame)
    assert result.summary.succeeded_rows == 4
    assert [row.record_id for row in result.rows] == [
        "row_000001",
        "row_000002",
        "row_000003",
        "row_000004",
    ]
    assert [row.batch_percentile for row in result.rows] == [62.5, 62.5, 100.0, 25.0]
    assert [row.top_10_percent_flag for row in result.rows] == [False, False, True, False]


def test_row_failures_do_not_block_valid_rows_and_missing_is_warning():
    frame = pd.DataFrame(
        {
            "record_id": ["ok", "bad-number", "bad-infinity", "too-many"],
            "f1": [2, "x", np.inf, np.nan],
            "f2": [1, 1, 1, np.nan],
            "f3": [0, 0, 0, np.nan],
        }
    )
    result = score_batch(make_active(), frame)
    assert result.summary.succeeded_rows == 2
    assert result.rows[0].warning_codes == ()
    assert result.rows[1].error_code == "NON_NUMERIC_VALUE"
    assert result.rows[2].error_code == "NON_FINITE_VALUE"
    assert result.rows[3].row_status is RowStatus.SUCCESS
    assert WarningCode.IMPUTED_VALUES in result.rows[3].warning_codes


def test_range_warning_is_consumed_without_changing_score():
    class Monitor:
        def flag_out_of_range(self, frame):
            return SimpleNamespace(warning_codes=((WarningCode.OUT_OF_RANGE.value,),))

    frame = pd.DataFrame({"f1": [8], "f2": [1], "f3": [0]})
    result = score_batch(make_active(Monitor()), frame)
    assert result.rows[0].confidence_score == 0.8
    assert result.rows[0].warning_codes == (WarningCode.OUT_OF_RANGE,)


def test_file_schema_error_does_not_score():
    frame = pd.DataFrame({"f1": [1], "f2": [1], "unknown": [1]})
    with pytest.raises(Exception):
        score_batch(make_active(), frame)


def test_all_failed_rows_skip_model_and_single_row_is_top_ten():
    frame = pd.DataFrame({"f1": ["x"], "f2": [1], "f3": [1]})
    result = score_batch(make_active(), frame)
    assert result.summary.succeeded_rows == 0
    assert result.rows[0].row_status is RowStatus.FAILED

    one = score_batch(make_active(), pd.DataFrame({"f1": [7], "f2": [1], "f3": [1]}))
    assert one.rows[0].top_10_percent_flag is True
    assert one.rows[0].batch_percentile == 100.0


def test_threshold_only_changes_risk_class():
    frame = pd.DataFrame({"f1": [6], "f2": [1], "f3": [1]})
    low = score_batch(make_active(), frame, threshold=0.5)
    high = score_batch(make_active(), frame, threshold=0.7)
    assert low.rows[0].risk_class != high.rows[0].risk_class
    assert low.rows[0].confidence_score == high.rows[0].confidence_score
    assert low.rows[0].batch_percentile == high.rows[0].batch_percentile
