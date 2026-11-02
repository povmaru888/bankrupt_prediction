"""Deterministic removal of only the approved constant and duplicate features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

CONSTANT_FEATURE = "Net Income Flag"
APPROVED_DUPLICATES = {
    "Current Liability to Liability": "Current Liabilities/Liability",
    "Current Liability to Equity": "Current Liabilities/Equity",
}
CORRELATION_THRESHOLD = 0.95


@dataclass(frozen=True)
class CorrelationPair:
    feature_a: str
    feature_b: str
    pearson_r: float


@dataclass(frozen=True)
class ExactDuplicatePair:
    first_feature: str
    duplicate_feature: str


@dataclass(frozen=True)
class SelectorAudit:
    removed_columns: tuple[str, ...]
    retained_columns: tuple[str, ...]
    correlation_threshold: float
    high_correlation_pairs: tuple[CorrelationPair, ...]
    unexpected_exact_duplicates: tuple[ExactDuplicatePair, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "removed_columns": list(self.removed_columns),
            "retained_columns": list(self.retained_columns),
            "correlation_threshold": self.correlation_threshold,
            "high_correlation_pairs": [asdict(pair) for pair in self.high_correlation_pairs],
            "unexpected_exact_duplicates": [
                asdict(pair) for pair in self.unexpected_exact_duplicates
            ],
        }


class FeatureSelector(TransformerMixin, BaseEstimator):
    """Drop the SDD-approved constant and exact duplicate fields after fitting."""

    def fit(self, X: Any, y: Any = None) -> FeatureSelector:
        frame = self._normalize_frame(X)
        missing = sorted(
            ({CONSTANT_FEATURE} | set(APPROVED_DUPLICATES) | set(APPROVED_DUPLICATES.values()))
            - set(frame.columns)
        )
        if missing:
            raise ValueError(f"training features are missing required cleanup columns: {missing}")
        if frame[CONSTANT_FEATURE].nunique(dropna=False) != 1:
            raise ValueError(
                f"{CONSTANT_FEATURE!r} is expected to be constant in the training data"
            )

        expected_pair_set = {
            frozenset((duplicate, original)) for duplicate, original in APPROVED_DUPLICATES.items()
        }
        approved_to_remove: set[str] = {CONSTANT_FEATURE}
        for duplicate, original in APPROVED_DUPLICATES.items():
            if not frame[duplicate].equals(frame[original]):
                raise ValueError(
                    f"approved duplicate {duplicate!r} no longer exactly matches {original!r}"
                )
            approved_to_remove.add(duplicate)

        unexpected_duplicates: list[ExactDuplicatePair] = []
        for first, second in combinations(frame.columns, 2):
            if (
                frame[first].equals(frame[second])
                and frozenset((first, second)) not in expected_pair_set
            ):
                unexpected_duplicates.append(ExactDuplicatePair(first, second))

        correlations = frame.corr(method="pearson", numeric_only=True)
        high_pairs = tuple(
            CorrelationPair(first, second, float(value))
            for first, second in combinations(frame.columns, 2)
            if pd.notna(value := correlations.loc[first, second])
            and abs(float(value)) >= CORRELATION_THRESHOLD
        )

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.removed_features_ = tuple(name for name in frame.columns if name in approved_to_remove)
        self.retained_features_ = tuple(
            name for name in frame.columns if name not in approved_to_remove
        )
        self.audit_ = SelectorAudit(
            removed_columns=self.removed_features_,
            retained_columns=self.retained_features_,
            correlation_threshold=CORRELATION_THRESHOLD,
            high_correlation_pairs=high_pairs,
            unexpected_exact_duplicates=tuple(unexpected_duplicates),
        )
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, attributes=["feature_names_in_", "retained_features_"])
        frame = self._normalize_frame(X)
        if tuple(frame.columns) != tuple(self.feature_names_in_):
            raise ValueError("input feature names or order differ from fitted training data")
        return frame.loc[:, list(self.retained_features_)].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, attributes=["retained_features_"])
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("input_features differ from fitted feature names")
        return np.asarray(self.retained_features_, dtype=object)

    @staticmethod
    def _normalize_frame(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSelector requires a pandas DataFrame with named columns")
        if any(not isinstance(name, str) for name in X.columns):
            raise ValueError("feature names must be strings")
        frame = X.copy()
        frame.columns = [name.strip() for name in frame.columns]
        if any(not name for name in frame.columns):
            raise ValueError("feature names cannot be empty")
        if frame.columns.has_duplicates:
            raise ValueError("feature names must be unique after trimming whitespace")
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError("all feature columns must be numeric")
        return frame


def selector_audit_to_dict(selector: FeatureSelector) -> dict[str, Any]:
    """Return the fitted selector audit in a manifest-friendly JSON shape."""
    check_is_fitted(selector, attributes=["audit_"])
    return selector.audit_.to_dict()
