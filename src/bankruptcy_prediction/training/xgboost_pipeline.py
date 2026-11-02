"""CPU XGBoost Pipeline with fold-local positive-class weighting."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from bankruptcy_prediction.features.imputation import build_imputer
from bankruptcy_prediction.features.selector import FeatureSelector


class FoldWeightedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible XGBoost wrapper deriving class weight inside each fit."""

    def __init__(
        self,
        *,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 300,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: str | float = "balanced",
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> FoldWeightedXGBClassifier:
        labels = np.asarray(y)
        if labels.ndim != 1 or set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("y must contain both binary labels 0 and 1")
        if self.scale_pos_weight == "balanced":
            negative_count = int(np.count_nonzero(labels == 0))
            positive_count = int(np.count_nonzero(labels == 1))
            if positive_count == 0:
                raise ValueError("training fold must contain positive examples")
            resolved_weight = negative_count / positive_count
        else:
            resolved_weight = float(self.scale_pos_weight)
            if not np.isfinite(resolved_weight) or resolved_weight <= 0:
                raise ValueError("scale_pos_weight must be 'balanced' or a positive finite number")

        self.classifier_ = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            random_state=self.random_state,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=resolved_weight,
        )
        self.classifier_.fit(X, labels)
        self.classes_ = self.classifier_.classes_
        self.n_features_in_ = self.classifier_.n_features_in_
        self.resolved_scale_pos_weight_ = resolved_weight
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes=["classifier_"])
        return self.classifier_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes=["classifier_"])
        return self.classifier_.predict_proba(X)


def build_xgboost_pipeline() -> Pipeline:
    """Build selector, fold-fitted median imputer, and CPU XGBoost classifier."""
    return Pipeline(
        steps=[
            ("selector", FeatureSelector()),
            ("imputer", build_imputer()),
            ("classifier", FoldWeightedXGBClassifier()),
        ]
    )


def xgboost_search_space() -> dict[str, Any]:
    """Return bounded estimator distributions; class balance is resolved per fit fold."""
    return {
        "classifier__max_depth": randint(3, 10),
        "classifier__learning_rate": loguniform(0.01, 0.3),
        "classifier__n_estimators": randint(100, 1001),
        "classifier__subsample": uniform(0.6, 0.4),
        "classifier__colsample_bytree": uniform(0.6, 0.4),
        "classifier__min_child_weight": randint(1, 11),
        "classifier__reg_alpha": loguniform(1e-8, 10.0),
        "classifier__reg_lambda": loguniform(1e-3, 100.0),
        "classifier__scale_pos_weight": ["balanced"],
    }
