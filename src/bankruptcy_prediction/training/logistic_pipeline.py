"""Leakage-safe Logistic Regression Pipeline and randomized search space."""

from __future__ import annotations

from typing import Any

from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from bankruptcy_prediction.features.imputation import build_imputer
from bankruptcy_prediction.features.selector import FeatureSelector

DEFAULT_MAX_ITER = 5000
C_MIN = 1e-4
C_MAX = 1e4


def build_logistic_pipeline(max_iter: int = DEFAULT_MAX_ITER) -> Pipeline:
    """Build all learnable preprocessing and classifier steps into one Pipeline."""
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    return Pipeline(
        steps=[
            ("selector", FeatureSelector()),
            ("imputer", build_imputer()),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="saga",
                    l1_ratio=0.0,
                    C=1.0,
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=42,
                ),
            ),
        ]
    )


def logistic_search_space() -> dict[str, Any]:
    """Return named randomized-search distributions compatible with saga L1/L2."""
    return {
        "scaler": [StandardScaler(), RobustScaler()],
        # sklearn 1.8+ maps saga l1_ratio=0 to L2 and l1_ratio=1 to L1.
        "classifier__l1_ratio": [0.0, 1.0],
        "classifier__C": loguniform(C_MIN, C_MAX),
        "classifier__class_weight": [None, "balanced"],
    }
