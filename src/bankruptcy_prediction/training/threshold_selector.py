"""Select a reproducible default classification threshold from validation data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_score, recall_score

F2_BETA = 2.0
F2_TIE_RULE = "lowest_threshold"
THRESHOLD_SELECTION_RULE = "validation_f2_maximum; lowest_threshold_on_tie"


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    precision: float
    recall: float
    f2: float


@dataclass(frozen=True)
class ThresholdResult:
    default_threshold: float
    best_f2: float
    curve_summary: tuple[ThresholdPoint, ...]
    rule: str = THRESHOLD_SELECTION_RULE

    def to_dict(self) -> dict[str, object]:
        return {
            "default_threshold": self.default_threshold,
            "best_f2": self.best_f2,
            "rule": self.rule,
            "curve_summary": [
                {
                    "threshold": point.threshold,
                    "precision": point.precision,
                    "recall": point.recall,
                    "f2": point.f2,
                }
                for point in self.curve_summary
            ],
        }


def select_f2_threshold(y_val: object, confidence_scores: object) -> ThresholdResult:
    """Choose the lowest validation threshold among those attaining maximum F2.

    The function accepts only validation labels and scores; final-test data is not
    an input. A prediction is positive when its score is greater than or equal to
    the threshold. Candidate thresholds include 0 and 1 as well as every unique
    observed score, so boundary behavior is explicit and reproducible.
    """
    labels = np.asarray(y_val)
    scores = np.asarray(confidence_scores, dtype=float)
    if labels.ndim != 1 or labels.size == 0 or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("y_val must be a non-empty one-dimensional array containing 0 and 1")
    if scores.ndim != 1 or len(scores) != len(labels):
        raise ValueError("confidence_scores must be one-dimensional and match y_val")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("confidence_scores must be finite values between 0 and 1")

    thresholds = np.unique(np.concatenate(([0.0], scores, [1.0])))
    points: list[ThresholdPoint] = []
    for threshold in thresholds:
        predictions = scores >= threshold
        precision = float(precision_score(labels, predictions, zero_division=0))
        recall = float(recall_score(labels, predictions, zero_division=0))
        denominator = 4 * precision + recall
        f2 = 0.0 if denominator == 0 else float(5 * precision * recall / denominator)
        points.append(
            ThresholdPoint(threshold=float(threshold), precision=precision, recall=recall, f2=f2)
        )

    best_f2 = max(point.f2 for point in points)
    # `thresholds` is ascending; first maximum implements the documented lower-threshold tie rule.
    best = next(point for point in points if point.f2 == best_f2)
    return ThresholdResult(
        default_threshold=best.threshold,
        best_f2=best.f2,
        curve_summary=tuple(points),
    )
