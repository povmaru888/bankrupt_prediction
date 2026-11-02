"""Training-only sigmoid calibration for candidate classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

CALIBRATION_FOLDS = 5
CALIBRATION_METHOD = "sigmoid"
RANDOM_SEED = 42


@dataclass(frozen=True)
class CalibrationMetadata:
    method: str
    cv_folds: int
    random_seed: int
    training_row_count: int


def calibrate_candidate(
    best_estimator: Any,
    X_train: Any,
    y_train: Any,
    *,
    cv_folds: int = CALIBRATION_FOLDS,
) -> CalibratedClassifierCV:
    """Fit a sigmoid calibrator on training data using configured folds (5 in production)."""
    labels = np.asarray(y_train)
    if labels.ndim != 1 or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("y_train must contain both binary labels 0 and 1")
    counts = np.unique(labels, return_counts=True)[1]
    if isinstance(cv_folds, bool) or not isinstance(cv_folds, int) or cv_folds < 2:
        raise ValueError("cv_folds must be an integer of at least two")
    if counts.min() < cv_folds:
        raise ValueError(f"each class needs at least {cv_folds} training rows for calibration")
    if len(X_train) != len(labels):
        raise ValueError("X_train and y_train lengths must match")

    calibrator = CalibratedClassifierCV(
        estimator=best_estimator,
        method=CALIBRATION_METHOD,
        cv=StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=RANDOM_SEED,
        ),
        n_jobs=1,
    )
    calibrator.fit(X_train, labels)
    calibrator.calibration_metadata_ = CalibrationMetadata(
        method=CALIBRATION_METHOD,
        cv_folds=cv_folds,
        random_seed=RANDOM_SEED,
        training_row_count=len(labels),
    )
    return calibrator
