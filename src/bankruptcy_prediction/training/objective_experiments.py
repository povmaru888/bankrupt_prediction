"""Experimental XGBoost objectives with a stable, serializable sklearn API."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBRanker, XGBRegressor

SEED = 42


class ObjectiveExperimentModel(ClassifierMixin, BaseEstimator):
    """Classifier-shaped adapter exposing monotonic scores for AP search."""

    def __init__(
        self,
        *,
        objective_kind: str,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 300,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        huber_slope: float = 1.0,
        lambdarank_num_pair_per_sample: int = 8,
        random_state: int = SEED,
    ) -> None:
        self.objective_kind = objective_kind
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.huber_slope = huber_slope
        self.lambdarank_num_pair_per_sample = lambdarank_num_pair_per_sample
        self.random_state = random_state

    def _common(self) -> dict[str, Any]:
        return {
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": 1,
            "random_state": self.random_state,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }

    def fit(self, X: Any, y: Any) -> ObjectiveExperimentModel:
        labels = np.asarray(y, dtype=int)
        if labels.ndim != 1 or set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("each fit must contain both binary labels")
        if self.objective_kind == "rank_pairwise":
            self.estimator_ = XGBRanker(
                objective="rank:pairwise",
                eval_metric="map",
                lambdarank_pair_method="mean",
                lambdarank_num_pair_per_sample=self.lambdarank_num_pair_per_sample,
                **self._common(),
            )
            self.estimator_.fit(X, labels, qid=np.zeros(len(labels), dtype=np.int32))
            self.weighting_ = "one_query_per_fit"
        elif self.objective_kind == "pseudo_huber":
            negatives = int(np.count_nonzero(labels == 0))
            positives = int(np.count_nonzero(labels == 1))
            positive_weight = negatives / positives
            weights = np.where(labels == 1, positive_weight, 1.0)
            self.estimator_ = XGBRegressor(
                objective="reg:pseudohubererror",
                eval_metric="mphe",
                huber_slope=self.huber_slope,
                **self._common(),
            )
            self.estimator_.fit(X, labels.astype(float), sample_weight=weights)
            self.weighting_ = {"positive_weight": positive_weight}
        else:
            raise ValueError(f"unsupported objective_kind: {self.objective_kind}")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator_.predict(X), dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        positive = expit(self.decision_function(X))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
