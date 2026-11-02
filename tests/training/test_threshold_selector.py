"""Validation-only F2 threshold selection tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import fbeta_score

from bankruptcy_prediction.training.threshold_selector import (
    THRESHOLD_SELECTION_RULE,
    select_f2_threshold,
)


def test_selects_maximum_f2_and_returns_curve_summary() -> None:
    labels = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([0.1, 0.8, 0.3, 0.4, 0.7, 0.2])

    result = select_f2_threshold(labels, scores)

    assert result.best_f2 == pytest.approx(
        fbeta_score(labels, scores >= result.default_threshold, beta=2)
    )
    assert result.best_f2 == max(point.f2 for point in result.curve_summary)
    assert 0 <= result.default_threshold <= 1
    assert result.rule == THRESHOLD_SELECTION_RULE
    assert result.to_dict()["curve_summary"]


def test_tied_f2_chooses_lowest_threshold_to_favor_recall() -> None:
    # Identical scores give the same F2 at all candidates; lower threshold is deterministic.
    result = select_f2_threshold([0, 1], [0.5, 0.5])

    assert result.default_threshold == 0.0
    assert result.curve_summary[0].recall == 1.0


def test_includes_zero_and_one_boundaries() -> None:
    result = select_f2_threshold([0, 1], [0.0, 1.0])

    assert result.curve_summary[0].threshold == 0.0
    assert result.curve_summary[-1].threshold == 1.0


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([0, 0], [0.2, 0.8]),
        ([0, 1], [0.2]),
        ([0, 1], [0.2, np.nan]),
        ([0, 1], [-0.1, 1.1]),
    ],
)
def test_rejects_invalid_inputs(labels: object, scores: object) -> None:
    with pytest.raises(ValueError):
        select_f2_threshold(labels, scores)


def test_threshold_selector_has_no_test_set_argument() -> None:
    # Test-set isolation is structural: the public API receives only validation inputs.
    import inspect

    assert tuple(inspect.signature(select_f2_threshold).parameters) == (
        "y_val",
        "confidence_scores",
    )
