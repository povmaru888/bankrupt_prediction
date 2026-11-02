"""Tests for Logistic Regression pipeline composition and search compatibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from bankruptcy_prediction.data.training_loader import load_training_data
from bankruptcy_prediction.features.imputation import TrainingMedianImputer
from bankruptcy_prediction.features.selector import FeatureSelector
from bankruptcy_prediction.training.logistic_pipeline import (
    C_MAX,
    C_MIN,
    DEFAULT_MAX_ITER,
    build_logistic_pipeline,
    logistic_search_space,
)

ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_contains_only_fold_fitted_transformers_and_logistic_classifier() -> None:
    pipeline = build_logistic_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["selector", "imputer", "scaler", "classifier"]
    assert isinstance(pipeline.named_steps["selector"], FeatureSelector)
    assert isinstance(pipeline.named_steps["imputer"], TrainingMedianImputer)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.solver == "saga"
    assert classifier.max_iter >= DEFAULT_MAX_ITER
    assert classifier.class_weight == "balanced"
    assert "smote" not in pipeline.named_steps


def test_search_space_contains_compatible_scaler_penalty_c_and_class_weights() -> None:
    pipeline = build_logistic_pipeline()
    search = logistic_search_space()

    assert set(search) == {
        "scaler",
        "classifier__l1_ratio",
        "classifier__C",
        "classifier__class_weight",
    }
    assert any(isinstance(scaler, StandardScaler) for scaler in search["scaler"])
    assert any(isinstance(scaler, RobustScaler) for scaler in search["scaler"])
    assert search["classifier__l1_ratio"] == [0.0, 1.0]
    assert search["classifier__class_weight"] == [None, "balanced"]
    assert C_MIN > 0 and C_MAX > C_MIN

    for l1_ratio in search["classifier__l1_ratio"]:
        candidate = pipeline.set_params(classifier__l1_ratio=l1_ratio)
        assert candidate.get_params()["classifier__solver"] == "saga"


def test_convergence_warning_is_not_silenced() -> None:
    data = load_training_data(ROOT / "Database" / "bankrupt_data.csv")
    X = data.features.iloc[:2]
    y = np.array([0, 1])
    pipeline = build_logistic_pipeline(max_iter=1)

    with pytest.warns(ConvergenceWarning):
        pipeline.fit(X, y)


@pytest.mark.parametrize("max_iter", [0, -1, True])
def test_rejects_invalid_iteration_limit(max_iter: int) -> None:
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        build_logistic_pipeline(max_iter=max_iter)
