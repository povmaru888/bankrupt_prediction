"""Median imputation fitted only on the training data supplied to ``fit``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from bankruptcy_prediction.contracts import WarningCode


class TrainingMedianImputer(TransformerMixin, BaseEstimator):
    """A cloneable sklearn transformer with a row-level warning helper."""

    def fit(self, X: Any, y: Any = None) -> TrainingMedianImputer:
        values = self._as_array(X)
        self._validate_values(values)
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("training data must contain rows and features")
        all_missing = np.isnan(values).all(axis=0)
        if all_missing.any():
            columns = np.flatnonzero(all_missing).tolist()
            raise ValueError(f"cannot estimate median for all-missing training features: {columns}")
        self.statistics_ = np.nanmedian(values, axis=0)
        self.n_features_in_ = values.shape[1]
        if isinstance(X, pd.DataFrame) and all(isinstance(name, str) for name in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes=["statistics_", "n_features_in_"])
        values = self._as_array(X)
        self._validate_values(values)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features; received {values.shape[1]}")
        if hasattr(self, "feature_names_in_") and isinstance(X, pd.DataFrame):
            if list(X.columns) != self.feature_names_in_.tolist():
                raise ValueError(
                    "input feature names or order differ from the fitted training data"
                )
        missing = np.isnan(values)
        return np.where(missing, self.statistics_, values)

    def transform_with_warnings(self, X: Any) -> tuple[np.ndarray, tuple[tuple[str, ...], ...]]:
        """Transform rows and attach ``IMPUTED_VALUES`` only where values were missing."""
        values = self._as_array(X)
        warnings = tuple(
            (WarningCode.IMPUTED_VALUES.value,) if row_missing.any() else ()
            for row_missing in np.isnan(values)
        )
        return self.transform(X), warnings

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, attributes=["n_features_in_"])
        if input_features is not None:
            names = np.asarray(input_features, dtype=object)
            if len(names) != self.n_features_in_:
                raise ValueError("input_features length does not match fitted feature count")
            if (
                hasattr(self, "feature_names_in_")
                and names.tolist() != self.feature_names_in_.tolist()
            ):
                raise ValueError("input_features differ from fitted feature names")
            return names
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_.copy()
        return np.asarray([f"x{index}" for index in range(self.n_features_in_)], dtype=object)

    @staticmethod
    def _as_array(X: Any) -> np.ndarray:
        try:
            values = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("imputer input must be numeric or missing") from exc
        if values.ndim != 2:
            raise ValueError("imputer input must be a two-dimensional array")
        return values

    @staticmethod
    def _validate_values(values: np.ndarray) -> None:
        if np.isinf(values).any():
            raise ValueError("imputer input cannot contain infinite values")


def build_imputer() -> TrainingMedianImputer:
    """Create an unfitted median imputer for a training Pipeline."""
    return TrainingMedianImputer()
