"""Feature-level SHAP explanations for Logistic Regression and XGBoost models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

SUPPORTED_MODEL_FAMILIES = frozenset({"logistic_regression", "xgboost"})
FACTORS_PER_DIRECTION = 5
EXPORT_FACTORS_PER_DIRECTION = 3


@dataclass(frozen=True)
class ExplanationFactor:
    feature_name: str
    feature_value: float | None
    contribution: float

    def to_dict(self) -> dict[str, str | float | None]:
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "contribution": self.contribution,
        }


@dataclass(frozen=True)
class ExplanationRow:
    positive_factors: tuple[ExplanationFactor, ...]
    negative_factors: tuple[ExplanationFactor, ...]

    def project(self, limit: int = EXPORT_FACTORS_PER_DIRECTION) -> ExplanationRow:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        return ExplanationRow(
            positive_factors=self.positive_factors[:limit],
            negative_factors=self.negative_factors[:limit],
        )

    def to_dict(self) -> dict[str, list[dict[str, str | float | None]]]:
        return {
            "positive_factors": [factor.to_dict() for factor in self.positive_factors],
            "negative_factors": [factor.to_dict() for factor in self.negative_factors],
        }


@dataclass(frozen=True)
class ExplanationBatch:
    rows: tuple[ExplanationRow, ...]
    feature_names: tuple[str, ...]
    model_family: str

    def project(self, limit: int = EXPORT_FACTORS_PER_DIRECTION) -> ExplanationBatch:
        return ExplanationBatch(
            rows=tuple(row.project(limit) for row in self.rows),
            feature_names=self.feature_names,
            model_family=self.model_family,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "feature_names": list(self.feature_names),
            "rows": [row.to_dict() for row in self.rows],
        }


def _shap_module() -> Any:
    import shap

    return shap


def _estimators(model: Any) -> tuple[Any, ...]:
    """Return fitted base estimators from an optional calibrated CV ensemble."""
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated is not None:
        estimators = tuple(item.estimator for item in calibrated)
        if not estimators:
            raise ValueError("calibrated model has no fitted base estimators")
        return estimators
    return (model,)


def _prepare(estimator: Any, X: pd.DataFrame) -> tuple[Any, np.ndarray, tuple[str, ...]]:
    if isinstance(estimator, Pipeline):
        values: Any = X
        names = tuple(str(name) for name in X.columns)
        steps = list(estimator.steps)
        for _, transformer in steps[:-1]:
            values = transformer.transform(values)
            if isinstance(values, pd.DataFrame):
                names = tuple(str(name) for name in values.columns)
            elif hasattr(transformer, "get_feature_names_out"):
                try:
                    names = tuple(str(name) for name in transformer.get_feature_names_out(names))
                except TypeError:
                    names = tuple(str(name) for name in transformer.get_feature_names_out())
        classifier = steps[-1][1]
        return classifier, np.asarray(values, dtype=float), names
    if hasattr(estimator, "coef_"):
        names = tuple(str(name) for name in getattr(estimator, "feature_names_in_", X.columns))
        return estimator, X.to_numpy(dtype=float), names
    raise TypeError("model must be a fitted sklearn Pipeline or fitted classifier")


def _tree_classifier(classifier: Any) -> Any:
    # The project wrapper stores the fitted XGBClassifier as classifier_.
    return getattr(classifier, "classifier_", classifier)


def _normalize_shap_values(values: Any, row_count: int, feature_count: int) -> np.ndarray:
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        if not values:
            raise ValueError("SHAP returned no class values")
        values = values[-1]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        # SHAP v0.51 may return rows x features x classes for binary classifiers.
        array = array[:, :, -1]
    if array.shape != (row_count, feature_count):
        raise ValueError(
            f"unexpected SHAP value shape {array.shape}; expected {(row_count, feature_count)}"
        )
    if not np.isfinite(array).all():
        raise ValueError("SHAP values must be finite")
    return array


def _linear_shap(classifier: Any, values: np.ndarray) -> np.ndarray:
    if not hasattr(classifier, "coef_"):
        raise TypeError("Logistic Regression model must expose fitted coef_ values")
    coefficients = np.asarray(classifier.coef_, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[0] != 1:
        raise ValueError("Linear SHAP currently supports binary Logistic Regression only")
    intercept = np.asarray(getattr(classifier, "intercept_", [0.0]), dtype=float).reshape(-1)
    # Scaled feature value zero is the fitted training center for both SDD scalers.
    # This avoids using the current uploaded batch as a changing SHAP background.
    background = np.zeros((1, values.shape[1]), dtype=float)
    explainer = _shap_module().LinearExplainer(
        (coefficients.reshape(-1), float(intercept[0])), background
    )
    return _normalize_shap_values(explainer.shap_values(values), len(values), values.shape[1])


def _tree_shap(classifier: Any, values: np.ndarray) -> np.ndarray:
    explainer = _shap_module().TreeExplainer(_tree_classifier(classifier))
    result = explainer.shap_values(values)
    return _normalize_shap_values(result, len(values), values.shape[1])


def explain(
    model_family: Literal["logistic_regression", "xgboost"] | str,
    model: Any,
    X: pd.DataFrame,
) -> ExplanationBatch:
    """Explain each row using positive-class SHAP contributions, not causal effects.

    For calibrated CV models, explanations are averaged over the fitted base
    estimators. Logistic Regression contributions are in centered linear-score
    space; XGBoost contributions use Tree SHAP. Raw input feature values are
    retained alongside transformed-model attributions.
    """
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"unsupported model_family: {model_family}")
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise TypeError("X must be a non-empty pandas DataFrame with named features")
    if X.columns.has_duplicates or any(
        not isinstance(name, str) or not name.strip() for name in X.columns
    ):
        raise ValueError("X feature names must be unique non-empty strings")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("X features must be numeric")

    per_estimator_values: list[np.ndarray] = []
    expected_names: tuple[str, ...] | None = None
    for estimator in _estimators(model):
        classifier, transformed, names = _prepare(estimator, X)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError("calibrated estimators have inconsistent feature names or order")
        if model_family == "logistic_regression":
            values = _linear_shap(classifier, transformed)
        else:
            values = _tree_shap(classifier, transformed)
        per_estimator_values.append(values)

    assert expected_names is not None
    contributions = np.mean(np.stack(per_estimator_values, axis=0), axis=0)
    rows: list[ExplanationRow] = []
    raw_values = X.to_numpy(dtype=float)
    input_positions = {name: index for index, name in enumerate(X.columns)}
    if any(name not in input_positions for name in expected_names):
        raise ValueError("fitted model features are not present in the original input")
    for row_index, row_contributions in enumerate(contributions):
        positive_indices = sorted(
            (index for index, value in enumerate(row_contributions) if value > 0),
            key=lambda index: (-row_contributions[index], index),
        )[:FACTORS_PER_DIRECTION]
        negative_indices = sorted(
            (index for index, value in enumerate(row_contributions) if value < 0),
            key=lambda index: (row_contributions[index], index),
        )[:FACTORS_PER_DIRECTION]

        def factor(index: int) -> ExplanationFactor:
            name = expected_names[index]
            raw = raw_values[row_index, input_positions[name]]
            return ExplanationFactor(
                feature_name=name,
                feature_value=float(raw) if np.isfinite(raw) else None,
                contribution=float(row_contributions[index]),
            )

        rows.append(
            ExplanationRow(
                positive_factors=tuple(factor(index) for index in positive_indices),
                negative_factors=tuple(factor(index) for index in negative_indices),
            )
        )
    return ExplanationBatch(
        rows=tuple(rows), feature_names=expected_names, model_family=model_family
    )
