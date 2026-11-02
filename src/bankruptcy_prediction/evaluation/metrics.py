"""Test-set metrics for a frozen binary classifier and fixed threshold."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

TOP_FRACTION = 0.10
MIN_AP_BASELINE_MULTIPLE = 2.0
MIN_RECALL_AT_TOP_10 = 0.50


@dataclass(frozen=True)
class MetricsReport:
    average_precision: float
    positive_prevalence: float
    ap_to_prevalence_ratio: float | None
    precision: float
    recall: float
    f1: float
    brier_score: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    false_positive_count: int
    recall_at_top_10_percent: float
    top_10_percent_row_count: int
    top_10_percent_positive_count: int
    sample_count: int
    threshold: float
    mvp_passed: bool
    mvp_failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "average_precision": self.average_precision,
            "positive_prevalence": self.positive_prevalence,
            "ap_to_prevalence_ratio": self.ap_to_prevalence_ratio,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "brier_score": self.brier_score,
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "false_positive_count": self.false_positive_count,
            "recall_at_top_10_percent": self.recall_at_top_10_percent,
            "top_10_percent_row_count": self.top_10_percent_row_count,
            "top_10_percent_positive_count": self.top_10_percent_positive_count,
            "sample_count": self.sample_count,
            "threshold": self.threshold,
            "mvp_passed": self.mvp_passed,
            "mvp_failure_reasons": list(self.mvp_failure_reasons),
        }


def _positive_class_scores(model: Any, X_test: Any) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("frozen model must implement predict_proba")
    probabilities = np.asarray(model.predict_proba(X_test), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("predict_proba must return two class columns")
    scores = probabilities[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("positive-class scores must be finite values between 0 and 1")
    return scores


def evaluate_frozen_model(
    model: Any, threshold: float, X_test: Any, y_test: object
) -> MetricsReport:
    """Evaluate an already-fitted model once on final test data; this function never fits."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("frozen model must implement predict_proba")
    probabilities = np.asarray(model.predict_proba(X_test), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("predict_proba must return two class columns")
    return evaluate_frozen_scores(threshold, y_test, probabilities[:, 1])


def evaluate_frozen_scores(
    threshold: float, y_test: object, confidence_scores: object
) -> MetricsReport:
    """Compute final-test metrics from one already-produced positive-class score vector."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite value between 0 and 1")
    labels = np.asarray(y_test)
    if labels.ndim != 1 or labels.size == 0 or not set(np.unique(labels).tolist()) <= {0, 1}:
        raise ValueError("y_test must be a non-empty one-dimensional binary array")
    scores = np.asarray(confidence_scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(labels):
        raise ValueError("confidence_scores must be one-dimensional and match y_test")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("confidence_scores must be finite values between 0 and 1")
    predictions = scores >= threshold
    positive_count = int(np.sum(labels == 1))
    prevalence = positive_count / len(labels)
    if positive_count:
        average_precision = float(average_precision_score(labels, scores))
        ap_ratio: float | None = average_precision / prevalence
    else:
        # AP is undefined without positives; represent it as 0 and fail the MVP gate.
        average_precision = 0.0
        ap_ratio = None

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    ordered_indices = np.argsort(-scores, kind="stable")
    top_count = ceil(len(labels) * TOP_FRACTION)
    top_indices = ordered_indices[:top_count]
    top_positive_count = int(np.sum(labels[top_indices] == 1))
    recall_at_top = top_positive_count / positive_count if positive_count else 0.0
    failures: list[str] = []
    if ap_ratio is None or ap_ratio < MIN_AP_BASELINE_MULTIPLE:
        failures.append("average_precision_below_2x_prevalence")
    if recall_at_top < MIN_RECALL_AT_TOP_10:
        failures.append("recall_at_top_10_percent_below_0.50")

    return MetricsReport(
        average_precision=average_precision,
        positive_prevalence=prevalence,
        ap_to_prevalence_ratio=ap_ratio,
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        brier_score=float(brier_score_loss(labels, scores)),
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        false_positive_count=int(matrix[0, 1]),
        recall_at_top_10_percent=float(recall_at_top),
        top_10_percent_row_count=top_count,
        top_10_percent_positive_count=top_positive_count,
        sample_count=len(labels),
        threshold=float(threshold),
        mvp_passed=not failures,
        mvp_failure_reasons=tuple(failures),
    )
