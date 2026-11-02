"""Tests for sigmoid calibration scope, cross-validation, and score range."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from bankruptcy_prediction.training import calibration
from bankruptcy_prediction.training.calibration import calibrate_candidate

ROOT = Path(__file__).resolve().parents[2]


class _CalibrationSpy:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)

    def fit(self, X, y):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self


def test_calibrator_uses_only_supplied_training_data_and_fixed_cv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_X = np.arange(40, dtype=float).reshape(20, 2)
    train_y = np.array([0, 1] * 10)
    validation_X = np.full((4, 2), 999.0)
    test_X = np.full((4, 2), -999.0)
    _CalibrationSpy.calls = []
    monkeypatch.setattr(calibration, "CalibratedClassifierCV", _CalibrationSpy)

    result = calibrate_candidate(object(), train_X, train_y)

    captured = _CalibrationSpy.calls[0]
    assert isinstance(result, _CalibrationSpy)
    assert captured["method"] == "sigmoid"
    assert captured["n_jobs"] == 1
    assert captured["cv"].n_splits == 5
    assert captured["cv"].shuffle is True
    np.testing.assert_array_equal(result.fit_X, train_X)
    np.testing.assert_array_equal(result.fit_y, train_y)
    assert not np.isin(validation_X, result.fit_X).any()
    assert not np.isin(test_X, result.fit_X).any()


def test_real_calibrator_emits_model_confidence_scores_between_zero_and_one() -> None:
    X_train = np.arange(200, dtype=float).reshape(100, 2)
    y_train = np.array([0, 1] * 50)
    calibrator = calibrate_candidate(LogisticRegression(max_iter=1000), X_train, y_train)

    scores = calibrator.predict_proba(X_train[:12])[:, 1]

    assert isinstance(calibrator, CalibratedClassifierCV)
    assert calibrator.calibration_metadata_.method == "sigmoid"
    assert calibrator.calibration_metadata_.cv_folds == 5
    assert calibrator.calibration_metadata_.training_row_count == 100
    assert np.all((scores >= 0) & (scores <= 1))


def test_explicit_test_only_fold_count_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0, 1] * 10)
    _CalibrationSpy.calls = []
    monkeypatch.setattr(calibration, "CalibratedClassifierCV", _CalibrationSpy)

    result = calibrate_candidate(object(), X, y, cv_folds=2)

    assert _CalibrationSpy.calls[0]["cv"].n_splits == 2
    assert result.calibration_metadata_.cv_folds == 2


@pytest.mark.parametrize("y", [[0, 1, 0, 1], [0] * 10])
def test_rejects_too_small_or_single_class_training_data(y: list[int]) -> None:
    X = np.arange(len(y), dtype=float).reshape(-1, 1)

    with pytest.raises(ValueError, match="binary labels|at least 5"):
        calibrate_candidate(LogisticRegression(), X, y)
