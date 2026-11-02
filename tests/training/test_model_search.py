"""Orchestration tests for candidate count, CV policy, ordering, and failure capture."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import FitFailedWarning

from bankruptcy_prediction.settings import load_settings
from bankruptcy_prediction.training import model_search

ROOT = Path(__file__).resolve().parents[2]


class _SearchStub:
    calls: list[dict[str, object]] = []
    fail_one_candidate = False

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)

    def fit(self, X, y):
        assert len(X) == len(y)
        count = self.kwargs["n_iter"]
        self.cv_results_ = {
            "params": [{"candidate": index} for index in range(count)],
            "mean_test_score": np.linspace(0.1, 0.9, count),
        }
        if self.fail_one_candidate:
            self.cv_results_["mean_test_score"][3] = np.nan
            warnings.warn("candidate 3 failed in a training fold", FitFailedWarning, stacklevel=2)
        self.best_score_ = 0.9
        self.best_params_ = {"candidate": count - 1}
        self.best_estimator_ = self.kwargs["estimator"]
        return self


@pytest.fixture
def training_config():
    return load_settings(ROOT / "config/training.yaml", ROOT / "config/service.yaml").training


def test_search_uses_two_models_thirty_candidates_five_folds_and_average_precision(
    monkeypatch: pytest.MonkeyPatch, training_config
) -> None:
    _SearchStub.calls = []
    _SearchStub.fail_one_candidate = False
    monkeypatch.setattr(model_search, "RandomizedSearchCV", _SearchStub)

    results = model_search.search_models(np.zeros((20, 2)), np.array([0, 1] * 10), training_config)

    assert [call["n_iter"] for call in _SearchStub.calls] == [30, 30]
    assert [call["scoring"] for call in _SearchStub.calls] == ["average_precision"] * 2
    assert [call["n_jobs"] for call in _SearchStub.calls] == [1, 1]
    assert all(call["cv"].n_splits == 5 for call in _SearchStub.calls)
    assert all(call["cv"].shuffle is True for call in _SearchStub.calls)
    assert {result.model_name for result in results.ranked_models} == {
        "logistic_regression",
        "xgboost",
    }
    assert results.ranked_models[0].best_score >= results.ranked_models[1].best_score
    assert all(result.attempted_candidates == 30 for result in results.ranked_models)


def test_test_only_mode_allows_explicit_quick_search_configuration(
    monkeypatch: pytest.MonkeyPatch, training_config
) -> None:
    from dataclasses import replace

    _SearchStub.calls = []
    _SearchStub.fail_one_candidate = False
    monkeypatch.setattr(model_search, "RandomizedSearchCV", _SearchStub)
    quick = replace(
        training_config,
        test_only=True,
        search=replace(training_config.search, n_iter=2, cv_folds=2),
    )

    results = model_search.search_models(
        np.zeros((20, 2)), np.array([0, 1] * 10), quick, test_only=True
    )

    assert [call["n_iter"] for call in _SearchStub.calls] == [2, 2]
    assert all(call["cv"].n_splits == 2 for call in _SearchStub.calls)
    assert all(result.attempted_candidates == 2 for result in results.ranked_models)


def test_partial_fit_failures_are_recorded_with_params_and_reason(
    monkeypatch: pytest.MonkeyPatch, training_config
) -> None:
    _SearchStub.calls = []
    _SearchStub.fail_one_candidate = True
    monkeypatch.setattr(model_search, "RandomizedSearchCV", _SearchStub)

    results = model_search.search_models(np.zeros((20, 2)), np.array([0, 1] * 10), training_config)

    for result in (results.logistic_regression, results.xgboost):
        assert len(result.failed_candidates) == 1
        assert result.failed_candidates[0].parameters == {"candidate": 3}
        assert "candidate 3 failed" in result.failed_candidates[0].reason
        assert result.best_score == 0.9


@pytest.mark.parametrize(
    "path,value,expected",
    [
        ("random_seed", 7, "random_seed must remain fixed at 42"),
        ("n_iter", 29, "search.n_iter must remain fixed at 30"),
        ("cv_folds", 4, "search.cv_folds must remain fixed at 5"),
        ("scoring", "f1", "search.scoring must remain 'average_precision'"),
    ],
)
def test_search_rejects_reduced_or_changed_baseline(
    monkeypatch: pytest.MonkeyPatch, training_config, path: str, value: object, expected: str
) -> None:
    section = training_config.search
    from dataclasses import replace

    if path == "random_seed":
        invalid_training_config = replace(training_config, random_seed=value)
    else:
        invalid_training_config = replace(training_config, search=replace(section, **{path: value}))
    monkeypatch.setattr(model_search, "RandomizedSearchCV", _SearchStub)

    with pytest.raises(ValueError, match=expected):
        model_search.search_models(
            np.zeros((20, 2)), np.array([0, 1] * 10), invalid_training_config
        )
