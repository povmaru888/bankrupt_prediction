"""Training-range monitoring without clipping finite inference values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from bankruptcy_prediction.contracts import WarningCode

MAX_REPORTED_COLUMNS_PER_ROW = 10


@dataclass(frozen=True)
class WarningMatrix:
    warning_codes: tuple[tuple[str, ...], ...]
    out_of_range_columns: tuple[tuple[str, ...], ...]
    out_of_range_count: pd.Series


class RangeMonitor(BaseEstimator):
    """Stores per-feature training bounds and reports finite values outside them."""

    def fit(self, X_train: Any) -> RangeMonitor:
        values = self._as_array(X_train)
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("training data must contain rows and features")
        if np.isinf(values).any():
            raise ValueError("training range data cannot contain infinite values")
        all_missing = np.isnan(values).all(axis=0)
        if all_missing.any():
            raise ValueError("cannot estimate ranges for all-missing training features")
        self.minimum_ = np.nanmin(values, axis=0)
        self.maximum_ = np.nanmax(values, axis=0)
        self.n_features_in_ = values.shape[1]
        if isinstance(X_train, pd.DataFrame) and all(
            isinstance(name, str) for name in X_train.columns
        ):
            self.feature_names_in_ = np.asarray(X_train.columns, dtype=object)
        else:
            self.feature_names_in_ = np.asarray(
                [f"x{index}" for index in range(self.n_features_in_)], dtype=object
            )
        return self

    def flag_out_of_range(self, X: Any) -> WarningMatrix:
        """Return row warnings and bounded feature summaries without changing values."""
        check_is_fitted(self, attributes=["minimum_", "maximum_", "n_features_in_"])
        values = self._as_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features; received {values.shape[1]}")
        if np.isinf(values).any():
            raise ValueError("range monitoring input cannot contain infinite values")
        if isinstance(X, pd.DataFrame) and list(X.columns) != self.feature_names_in_.tolist():
            raise ValueError("input feature names or order differ from fitted training data")

        mask = (values < self.minimum_) | (values > self.maximum_)
        mask &= ~np.isnan(values)
        names = self.feature_names_in_.tolist()
        counts = mask.sum(axis=1).astype("int64")
        summaries = tuple(
            tuple(
                names[position] for position in np.flatnonzero(row)[:MAX_REPORTED_COLUMNS_PER_ROW]
            )
            for row in mask
        )
        warnings = tuple((WarningCode.OUT_OF_RANGE.value,) if count else () for count in counts)
        return WarningMatrix(
            warning_codes=warnings,
            out_of_range_columns=summaries,
            out_of_range_count=pd.Series(
                counts, index=getattr(X, "index", None), name="out_of_range_count"
            ),
        )

    @staticmethod
    def _as_array(X: Any) -> np.ndarray:
        try:
            values = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("range monitor input must be numeric") from exc
        if values.ndim != 2:
            raise ValueError("range monitor input must be two-dimensional")
        return values


def fit_ranges(X_train: Any) -> RangeMonitor:
    """Fit and return a range monitor using training data only."""
    return RangeMonitor().fit(X_train)
