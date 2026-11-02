"""Tests for final-test metrics and stable Top 10% recall evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from bankruptcy_prediction.evaluation.metrics import evaluate_frozen_model


class _FrozenModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=float)
        self.predict_calls = 0
        self.fit_calls = 0

    def predict_proba(self, X: object) -> np.ndarray:
        self.predict_calls += 1
        return np.column_stack([1 - self.scores, self.scores])

    def fit(self, X: object, y: object) -> None:
        self.fit_calls += 1


def test_t07_reports_metrics_and_stable_top_ten_percent() -> None:
    # T07: for 11 rows, ceil(11 * 10%) is 2; an equal-score boundary keeps earlier row.
    labels = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    scores = [0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    model = _FrozenModel(scores)

    report = evaluate_frozen_model(model, 0.75, np.zeros((11, 1)), labels)

    assert report.average_precision == pytest.approx(0.3353535354)
    assert report.confusion_matrix == ((6, 2), (2, 1))
    assert report.false_positive_count == 2
    assert report.top_10_percent_row_count == 2
    assert report.top_10_percent_positive_count == 1
    assert report.recall_at_top_10_percent == pytest.approx(1 / 3)
    assert report.brier_score >= 0
    assert model.predict_calls == 1 and model.fit_calls == 0


def test_equal_score_boundary_preserves_original_row_order() -> None:
    labels = [1, 0, 1, 0]
    model = _FrozenModel([0.9, 0.8, 0.8, 0.1])

    report = evaluate_frozen_model(model, 0.5, np.zeros((4, 1)), labels)

    assert report.top_10_percent_row_count == 1
    assert report.top_10_percent_positive_count == 1
    assert report.recall_at_top_10_percent == 0.5


def test_empty_positive_class_is_safe_and_fails_mvp_gate() -> None:
    report = evaluate_frozen_model(_FrozenModel([0.1, 0.2]), 0.5, np.zeros((2, 1)), [0, 0])

    assert report.average_precision == 0.0
    assert report.ap_to_prevalence_ratio is None
    assert report.recall_at_top_10_percent == 0.0
    assert not report.mvp_passed
    assert report.mvp_failure_reasons == (
        "average_precision_below_2x_prevalence",
        "recall_at_top_10_percent_below_0.50",
    )


def test_mvp_gate_requires_both_quality_conditions() -> None:
    labels = [0] * 18 + [1, 1]
    # Two positives ranked first: AP=1 (25x prevalence), Recall@Top10%=1.
    passed = evaluate_frozen_model(
        _FrozenModel([0.1] * 18 + [0.99, 0.98]), 0.5, np.zeros((20, 1)), labels
    )
    # Positives scored poorly: both criteria fail even if threshold is changed.
    failed = evaluate_frozen_model(
        _FrozenModel([0.99, 0.98] + [0.1] * 18), 0.5, np.zeros((20, 1)), labels
    )

    assert passed.mvp_passed
    assert not failed.mvp_passed


@pytest.mark.parametrize(
    "threshold,labels,scores",
    [
        (1.1, [0, 1], [0.1, 0.9]),
        (0.5, [], []),
        (0.5, [0, 2], [0.1, 0.9]),
        (0.5, [0, 1], [0.1]),
        (0.5, [0, 1], [0.1, np.inf]),
    ],
)
def test_rejects_invalid_evaluation_inputs(
    threshold: float, labels: list[int], scores: list[float]
) -> None:
    with pytest.raises(ValueError):
        evaluate_frozen_model(_FrozenModel(scores), threshold, np.zeros((len(scores), 1)), labels)
