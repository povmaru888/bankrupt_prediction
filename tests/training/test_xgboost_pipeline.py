"""Tests for CPU XGBoost parameters, search bounds, and fold-local weights."""

from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

from bankruptcy_prediction.features.imputation import TrainingMedianImputer
from bankruptcy_prediction.features.selector import FeatureSelector
from bankruptcy_prediction.training.xgboost_pipeline import (
    FoldWeightedXGBClassifier,
    build_xgboost_pipeline,
    xgboost_search_space,
)


def test_pipeline_has_selector_imputer_and_cpu_classifier() -> None:
    pipeline = build_xgboost_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["selector", "imputer", "classifier"]
    assert isinstance(pipeline.named_steps["selector"], FeatureSelector)
    assert isinstance(pipeline.named_steps["imputer"], TrainingMedianImputer)
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, FoldWeightedXGBClassifier)
    params = classifier.get_params()
    assert params["scale_pos_weight"] == "balanced"
    assert params["random_state"] == 42
    assert isinstance(clone(pipeline), Pipeline)


def test_search_space_contains_required_parameters_with_bounded_ranges() -> None:
    space = xgboost_search_space()
    required = {
        "classifier__max_depth",
        "classifier__learning_rate",
        "classifier__n_estimators",
        "classifier__subsample",
        "classifier__colsample_bytree",
        "classifier__min_child_weight",
        "classifier__reg_alpha",
        "classifier__reg_lambda",
        "classifier__scale_pos_weight",
    }
    assert set(space) == required
    assert isinstance(space["classifier__max_depth"], randint(1, 2).__class__)
    assert isinstance(space["classifier__learning_rate"], loguniform(1, 2).__class__)
    assert isinstance(space["classifier__subsample"], uniform(0, 1).__class__)
    assert space["classifier__scale_pos_weight"] == ["balanced"]

    samples = list(ParameterSampler(space, n_iter=12, random_state=42))
    for sample in samples:
        assert 3 <= sample["classifier__max_depth"] <= 9
        assert 0.01 <= sample["classifier__learning_rate"] <= 0.3
        assert 100 <= sample["classifier__n_estimators"] <= 1000
        assert 0.6 <= sample["classifier__subsample"] <= 1.0
        assert 0.6 <= sample["classifier__colsample_bytree"] <= 1.0
        assert 1 <= sample["classifier__min_child_weight"] <= 10
        assert 1e-8 <= sample["classifier__reg_alpha"] <= 10.0
        assert 1e-3 <= sample["classifier__reg_lambda"] <= 100.0
        assert sample["classifier__scale_pos_weight"] == "balanced"


def test_scale_pos_weight_uses_only_labels_passed_to_each_fit() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    first_fold_y = np.array([0, 0, 0, 0, 1])
    second_fold_y = np.array([0, 1, 1, 1, 1])
    first = FoldWeightedXGBClassifier(n_estimators=2, max_depth=2).fit(X, first_fold_y)
    second = FoldWeightedXGBClassifier(n_estimators=2, max_depth=2).fit(X, second_fold_y)

    assert first.resolved_scale_pos_weight_ == 4.0
    assert second.resolved_scale_pos_weight_ == 0.25
    assert first.classifier_.get_params()["device"] == "cpu"
    assert first.classifier_.get_params()["tree_method"] == "hist"
