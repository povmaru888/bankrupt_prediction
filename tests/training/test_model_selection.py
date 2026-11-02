"""Tests for validation AP ranking and the sub-0.01 Logistic tie-break."""

from __future__ import annotations

import numpy as np
import pytest

from bankruptcy_prediction.training import model_selection
from bankruptcy_prediction.training.model_selection import (
    LOGISTIC_NAME,
    SELECTION_RULE,
    XGBOOST_NAME,
    select_model,
)


class _ProbabilityEstimator:
    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        return np.column_stack([np.full(len(X), 1 - self.score), np.full(len(X), self.score)])


@pytest.mark.parametrize(
    "logistic_score,xgboost_score,expected",
    [
        (0.70, 0.80, XGBOOST_NAME),
        (0.60, 0.61, XGBOOST_NAME),
        (0.60, 0.609, LOGISTIC_NAME),
        (0.61, 0.60, LOGISTIC_NAME),
    ],
)
def test_selection_uses_validation_ap_and_fixed_boundary(
    monkeypatch: pytest.MonkeyPatch,
    logistic_score: float,
    xgboost_score: float,
    expected: str,
) -> None:
    scores = iter([logistic_score, xgboost_score])
    monkeypatch.setattr(model_selection, "average_precision_score", lambda y, p: next(scores))
    logistic = _ProbabilityEstimator(0.4)
    xgboost = _ProbabilityEstimator(0.5)
    candidates = {LOGISTIC_NAME: logistic, XGBOOST_NAME: xgboost}
    y_val = np.array([0, 1, 0, 1])

    selected = select_model(candidates, np.zeros((4, 2)), y_val)

    assert selected.model_name == expected
    assert selected.score_gap == abs(logistic_score - xgboost_score)
    assert selected.rule == SELECTION_RULE
    assert selected.to_dict()["validation_average_precision"] == {
        LOGISTIC_NAME: logistic_score,
        XGBOOST_NAME: xgboost_score,
    }
    assert logistic.calls == 1 and xgboost.calls == 1


def test_selection_rejects_incomplete_candidate_set() -> None:
    with pytest.raises(ValueError, match="must contain logistic_regression and xgboost"):
        select_model({LOGISTIC_NAME: _ProbabilityEstimator(0.5)}, np.zeros((2, 1)), [0, 1])


def test_selection_rejects_invalid_confidence_scores() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        select_model(
            {LOGISTIC_NAME: _ProbabilityEstimator(1.2), XGBOOST_NAME: _ProbabilityEstimator(0.5)},
            np.zeros((2, 1)),
            [0, 1],
        )
