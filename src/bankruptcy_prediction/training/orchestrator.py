"""End-to-end offline training orchestration, ending at a candidate package."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from bankruptcy_prediction.data.feature_schema import build_feature_schema
from bankruptcy_prediction.data.training_loader import load_training_data
from bankruptcy_prediction.evaluation.metrics import MetricsReport, evaluate_frozen_scores
from bankruptcy_prediction.explanation.shap_service import explain
from bankruptcy_prediction.model_package.builder import (
    CandidatePackageContext,
    ModelPackage,
    build_candidate_package,
)
from bankruptcy_prediction.reporting.model_report import ModelReportContext
from bankruptcy_prediction.settings import TrainingSettings
from bankruptcy_prediction.training.calibration import calibrate_candidate
from bankruptcy_prediction.training.model_search import ModelSearchResult, search_models
from bankruptcy_prediction.training.model_selection import (
    LOGISTIC_NAME,
    XGBOOST_NAME,
    select_model,
)
from bankruptcy_prediction.training.split_manager import create_split
from bankruptcy_prediction.training.threshold_selector import select_f2_threshold

FIXED_SPLIT = {"train": 0.64, "validation": 0.16, "test": 0.20}
VERSIONED_PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "shap", "joblib")


@dataclass(frozen=True)
class TrainingOutcome:
    package: ModelPackage
    model_family: str
    metrics: MetricsReport
    test_only: bool

    @property
    def quality_gate_passed(self) -> bool:
        return not self.test_only and self.metrics.mvp_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": (
                "test_only_completed"
                if self.test_only
                else "quality_gate_passed"
                if self.quality_gate_passed
                else "quality_gate_failed"
            ),
            "test_only": self.test_only,
            "model_version": self.package.version,
            "model_path": str(self.package.path),
            "model_family": self.model_family,
            "quality_gate_passed": self.quality_gate_passed,
            "observed_metrics_meet_mvp_thresholds": self.metrics.mvp_passed,
            "metrics": self.metrics.to_dict(),
        }


def _validate_training_settings(config: TrainingSettings) -> None:
    if config.random_seed != 42:
        raise ValueError("training random_seed must remain fixed at 42")
    actual_split = {
        "train": config.split.train,
        "validation": config.split.validation,
        "test": config.split.test,
    }
    if actual_split != FIXED_SPLIT:
        raise ValueError("training split must remain stratified 64% / 16% / 20%")
    if config.calibration.method != "sigmoid":
        raise ValueError("calibration.method must remain 'sigmoid'")
    if config.test_only:
        if (
            config.search.n_iter < 1
            or config.search.cv_folds < 2
            or config.calibration.cv_folds < 2
        ):
            raise ValueError("test-only iterations must be positive and folds must be at least two")
    elif (
        config.search.n_iter != 30
        or config.search.cv_folds != 5
        or config.calibration.cv_folds != 5
    ):
        raise ValueError(
            "production training requires 30 search candidates and 5-fold search/calibration"
        )


def _positive_scores(model: Any, X: Any) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("selected model predict_proba must return two class columns")
    scores = probabilities[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("selected model scores must be finite and between 0 and 1")
    return scores


def _search_summary(result: ModelSearchResult) -> dict[str, Any]:
    return {
        "model_name": result.model_name,
        "attempted_candidates": result.attempted_candidates,
        "best_validation_average_precision": result.best_score,
        "best_parameters": result.best_parameters,
        "failed_candidate_count": len(result.failed_candidates),
        "failure_reason": result.failure_reason,
    }


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in VERSIONED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def run_training(
    data_path: str | Path,
    config: TrainingSettings,
    models_root: str | Path,
) -> TrainingOutcome:
    """Train, select, threshold, evaluate once, and publish an immutable candidate."""
    _validate_training_settings(config)
    dataset = load_training_data(data_path, test_only=config.test_only)
    # Normalize once before every split, so SHAP and the package share input names.
    schema = build_feature_schema(dataset)
    dataset.features.columns = [field.name for field in schema.features]
    split = create_split(dataset.labels.to_numpy(), seed=config.random_seed)
    X_train = dataset.features.iloc[list(split.train_indices)]
    y_train = dataset.labels.iloc[list(split.train_indices)]
    X_validation = dataset.features.iloc[list(split.validation_indices)]
    y_validation = dataset.labels.iloc[list(split.validation_indices)]
    X_test = dataset.features.iloc[list(split.test_indices)]
    y_test = dataset.labels.iloc[list(split.test_indices)]

    searches = search_models(X_train, y_train, config, test_only=config.test_only)
    search_results = {
        LOGISTIC_NAME: searches.logistic_regression,
        XGBOOST_NAME: searches.xgboost,
    }
    candidates: dict[str, Any] = {}
    for name, result in search_results.items():
        if result.best_estimator is None or result.best_score is None:
            raise RuntimeError(f"{name} search did not produce a valid candidate")
        candidates[name] = calibrate_candidate(
            result.best_estimator,
            X_train,
            y_train,
            cv_folds=config.calibration.cv_folds,
        )

    selected = select_model(candidates, X_validation, y_validation)
    validation_scores = _positive_scores(selected.estimator, X_validation)
    threshold_result = select_f2_threshold(y_validation, validation_scores)

    # Final test scores are generated once and shared with metrics and the report.
    test_scores = _positive_scores(selected.estimator, X_test)
    metrics = evaluate_frozen_scores(threshold_result.default_threshold, y_test, test_scores)
    explanation = explain(selected.model_name, selected.estimator, X_validation)
    mean_abs_shap = {
        feature_name: float(
            np.mean(
                [
                    abs(
                        next(
                            (
                                factor.contribution
                                for factor in (*row.positive_factors, *row.negative_factors)
                                if factor.feature_name == feature_name
                            ),
                            0.0,
                        )
                    )
                    for row in explanation.rows
                ]
            )
        )
        for feature_name in explanation.feature_names
    }

    selected_search = search_results[selected.model_name]
    representative = selected.estimator.calibrated_classifiers_[0].estimator
    selector = representative.named_steps["selector"]
    split_dict = split.to_dict()
    calibration_metadata = asdict(selected.estimator.calibration_metadata_)
    metrics_dict = metrics.to_dict()
    report_context = ModelReportContext(
        dataset_audit=asdict(dataset.audit),
        split_summary={
            "seed": split.random_seed,
            "proportions": split_dict["proportions"],
            "index_sha256": split_dict["index_sha256"],
            "test_only": config.test_only,
        },
        search_summary={
            "logistic_regression": _search_summary(searches.logistic_regression),
            "xgboost": _search_summary(searches.xgboost),
            "test_only": config.test_only,
        },
        selection_summary=selected.to_dict(),
        threshold_summary=threshold_result.to_dict(),
        metrics_report=metrics_dict,
        y_test=y_test.to_numpy(),
        confidence_scores=test_scores,
        mean_abs_shap=mean_abs_shap,
        model_family=selected.model_name,
    )
    package_context = CandidatePackageContext(
        model=selected.estimator,
        model_family=selected.model_name,
        data_sha256=dataset.audit.source_sha256,
        split_seed=split.random_seed,
        split_indices={
            "train": split.train_indices,
            "validation": split.validation_indices,
            "test": split.test_indices,
        },
        feature_names=tuple(str(name) for name in dataset.features.columns),
        removed_columns=tuple(selector.removed_features_),
        package_versions=_package_versions(),
        model_parameters=selected_search.best_parameters or {},
        calibration=calibration_metadata,
        threshold_result=threshold_result,
        metrics_report=metrics_dict,
        report_context=report_context,
        test_only=config.test_only,
    )
    package = build_candidate_package(package_context, models_root)
    return TrainingOutcome(
        package=package,
        model_family=selected.model_name,
        metrics=metrics,
        test_only=config.test_only,
    )
