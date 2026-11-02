"""Randomized hyperparameter searches for both MVP candidate models."""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from bankruptcy_prediction.training.logistic_pipeline import (
    build_logistic_pipeline,
    logistic_search_space,
)
from bankruptcy_prediction.training.xgboost_pipeline import (
    build_xgboost_pipeline,
    xgboost_search_space,
)

EXPECTED_ITERATIONS = 30
EXPECTED_FOLDS = 5
EXPECTED_SCORING = "average_precision"
EXPECTED_SEED = 42


@dataclass(frozen=True)
class CandidateFailure:
    parameters: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class ModelSearchResult:
    model_name: str
    attempted_candidates: int
    best_score: float | None
    best_parameters: dict[str, Any] | None
    best_estimator: Any | None
    failed_candidates: tuple[CandidateFailure, ...] = ()
    failure_reason: str | None = None


@dataclass(frozen=True)
class SearchResults:
    logistic_regression: ModelSearchResult
    xgboost: ModelSearchResult

    @property
    def ranked_models(self) -> tuple[ModelSearchResult, ...]:
        results = (self.logistic_regression, self.xgboost)
        return tuple(
            sorted(
                results,
                key=lambda result: (
                    result.best_score is not None and np.isfinite(result.best_score),
                    result.best_score if result.best_score is not None else float("-inf"),
                ),
                reverse=True,
            )
        )


def _run_search(
    *,
    model_name: str,
    estimator: Any,
    parameter_space: dict[str, Any],
    X_train: Any,
    y_train: Any,
    config: Any,
) -> ModelSearchResult:
    cv = StratifiedKFold(
        n_splits=config.search.cv_folds,
        shuffle=True,
        random_state=config.random_seed,
    )
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_space,
        n_iter=config.search.n_iter,
        scoring=config.search.scoring,
        n_jobs=1,
        cv=cv,
        random_state=config.random_seed,
        refit=True,
        error_score=np.nan,
        return_train_score=False,
        verbose=2 if os.environ.get("BANKRUPTCY_TRAIN_VERBOSE") == "1" else 0,
    )
    captured_fit_failures: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FitFailedWarning)
            search.fit(X_train, y_train)
        captured_fit_failures = [
            str(item.message) for item in caught if issubclass(item.category, FitFailedWarning)
        ]
        if os.environ.get("BANKRUPTCY_TRAIN_VERBOSE") == "1":
            warning_counts: dict[str, int] = {}
            for item in caught:
                key = f"{item.category.__name__}: {item.message}"
                warning_counts[key] = warning_counts.get(key, 0) + 1
            for message, count in warning_counts.items():
                print(f"{model_name}: {count} warning(s): {message}", file=sys.stderr, flush=True)
    except ValueError as exc:
        if "fits failed" not in str(exc).lower():
            raise
        return ModelSearchResult(
            model_name=model_name,
            attempted_candidates=config.search.n_iter,
            best_score=None,
            best_parameters=None,
            best_estimator=None,
            failure_reason=str(exc),
        )

    scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    parameters = list(search.cv_results_["params"])
    failed_indices = np.flatnonzero(~np.isfinite(scores))
    reason = "; ".join(captured_fit_failures) or "candidate produced a non-finite CV score"
    failures = tuple(
        CandidateFailure(parameters=parameters[index], reason=reason) for index in failed_indices
    )
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return ModelSearchResult(
            model_name=model_name,
            attempted_candidates=len(parameters),
            best_score=None,
            best_parameters=None,
            best_estimator=None,
            failed_candidates=failures,
            failure_reason=reason,
        )
    return ModelSearchResult(
        model_name=model_name,
        attempted_candidates=len(parameters),
        best_score=float(search.best_score_),
        best_parameters=dict(search.best_params_),
        best_estimator=search.best_estimator_,
        failed_candidates=failures,
    )


def search_models(
    X_train: Any, y_train: Any, config: Any, *, test_only: bool = False
) -> SearchResults:
    """Search both models; reduced iterations/folds require an explicit test-only config."""
    test_only = test_only or bool(getattr(config, "test_only", False))
    if config.random_seed != EXPECTED_SEED:
        raise ValueError(f"random_seed must remain fixed at {EXPECTED_SEED}")
    if not test_only and config.search.n_iter != EXPECTED_ITERATIONS:
        raise ValueError(f"search.n_iter must remain fixed at {EXPECTED_ITERATIONS}")
    if not test_only and config.search.cv_folds != EXPECTED_FOLDS:
        raise ValueError(f"search.cv_folds must remain fixed at {EXPECTED_FOLDS}")
    if config.search.scoring != EXPECTED_SCORING:
        raise ValueError(f"search.scoring must remain {EXPECTED_SCORING!r}")
    if config.search.n_iter < 1 or config.search.cv_folds < 2:
        raise ValueError("search iterations must be positive and CV folds at least two")

    logistic = _run_search(
        model_name="logistic_regression",
        estimator=build_logistic_pipeline(),
        parameter_space=logistic_search_space(),
        X_train=X_train,
        y_train=y_train,
        config=config,
    )
    xgboost = _run_search(
        model_name="xgboost",
        estimator=build_xgboost_pipeline(),
        parameter_space=xgboost_search_space(),
        X_train=X_train,
        y_train=y_train,
        config=config,
    )
    return SearchResults(logistic_regression=logistic, xgboost=xgboost)
