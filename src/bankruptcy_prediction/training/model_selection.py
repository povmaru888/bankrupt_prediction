"""Validation-only candidate comparison with the fixed Logistic tie-break rule."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

TIE_DELTA = 0.01
LOGISTIC_NAME = "logistic_regression"
XGBOOST_NAME = "xgboost"
SELECTION_RULE = "validation_average_precision; logistic_regression_if_gap_below_0.01"


@dataclass(frozen=True)
class SelectedModel:
    model_name: str
    estimator: Any
    logistic_average_precision: float
    xgboost_average_precision: float
    score_gap: float
    rule: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "validation_average_precision": {
                LOGISTIC_NAME: self.logistic_average_precision,
                XGBOOST_NAME: self.xgboost_average_precision,
            },
            "score_gap": self.score_gap,
            "rule": self.rule,
            "reason": self.reason,
        }


def _confidence_scores(estimator: Any, X_val: Any) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("candidate estimator must implement predict_proba")
    probabilities = np.asarray(estimator.predict_proba(X_val), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("candidate predict_proba must return two class columns")
    scores = probabilities[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("candidate confidence scores must be finite values between 0 and 1")
    return scores


def select_model(candidates: Mapping[str, Any], X_val: Any, y_val: Any) -> SelectedModel:
    """Select a calibrated estimator using validation AP; never accepts final-test data."""
    if set(candidates) != {LOGISTIC_NAME, XGBOOST_NAME}:
        raise ValueError("candidates must contain logistic_regression and xgboost")
    labels = np.asarray(y_val)
    if labels.ndim != 1 or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("y_val must contain both binary labels 0 and 1")
    if len(X_val) != len(labels):
        raise ValueError("X_val and y_val lengths must match")

    logistic_score = float(
        average_precision_score(labels, _confidence_scores(candidates[LOGISTIC_NAME], X_val))
    )
    xgboost_score = float(
        average_precision_score(labels, _confidence_scores(candidates[XGBOOST_NAME], X_val))
    )
    gap = abs(logistic_score - xgboost_score)
    is_boundary = isclose(gap, TIE_DELTA, rel_tol=0.0, abs_tol=1e-12)
    if gap < TIE_DELTA and not is_boundary:
        winner = LOGISTIC_NAME
        reason = (
            f"Validation AP gap {gap:.12g} is below {TIE_DELTA:.2f}; "
            "Logistic Regression selected by the tie-break rule."
        )
    elif logistic_score >= xgboost_score:
        winner = LOGISTIC_NAME
        reason = f"Logistic Regression has the higher validation AP ({logistic_score:.6f})."
    else:
        winner = XGBOOST_NAME
        reason = f"XGBoost has the higher validation AP ({xgboost_score:.6f})."
    return SelectedModel(
        model_name=winner,
        estimator=candidates[winner],
        logistic_average_precision=logistic_score,
        xgboost_average_precision=xgboost_score,
        score_gap=gap,
        rule=SELECTION_RULE,
        reason=reason,
    )
