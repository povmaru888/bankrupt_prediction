"""Controlled XGBoost class-weight ablation on the frozen project split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from bankruptcy_prediction.data.feature_schema import build_feature_schema
from bankruptcy_prediction.data.training_loader import load_training_data
from bankruptcy_prediction.evaluation.metrics import evaluate_frozen_scores
from bankruptcy_prediction.model_package.verifier import verify_package
from bankruptcy_prediction.training.calibration import calibrate_candidate
from bankruptcy_prediction.training.split_manager import create_split
from bankruptcy_prediction.training.threshold_selector import select_f2_threshold
from bankruptcy_prediction.training.xgboost_pipeline import build_xgboost_pipeline


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_scores(model: Any, features: Any) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("model predict_proba must return two columns")
    scores = probabilities[:, 1]
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("positive-class scores must be finite and between zero and one")
    return scores


def controlled_unweighted_parameters(model_parameters: dict[str, Any]) -> dict[str, Any]:
    """Keep the frozen XGBoost parameters and change only class weighting."""
    parameters = dict(model_parameters)
    if parameters.get("classifier__scale_pos_weight") != "balanced":
        raise ValueError("source candidate is not the expected fold-balanced XGBoost model")
    parameters["classifier__scale_pos_weight"] = 1.0
    return parameters


def _resolved_weights(calibrated: Any) -> list[float]:
    weights: list[float] = []
    for item in calibrated.calibrated_classifiers_:
        estimator = getattr(item, "estimator", None)
        if estimator is None:
            estimator = getattr(item, "base_estimator", None)
        classifier = estimator.named_steps["classifier"]
        weights.append(float(classifier.resolved_scale_pos_weight_))
    return weights


def evaluate_class_weight_comparison(
    data_path: str | Path,
    candidate_dir: str | Path,
    model_output: str | Path,
) -> dict[str, Any]:
    """Compare balanced and unweighted variants without changing the active model."""
    candidate = Path(candidate_dir)
    verified = verify_package(candidate)
    manifest = _read_json(candidate / "manifest.json")
    source_metrics = _read_json(candidate / "metrics.json")

    dataset = load_training_data(data_path, test_only=False)
    dataset.features.columns = [field.name for field in build_feature_schema(dataset).features]
    if dataset.audit.source_sha256 != manifest.get("data_sha256"):
        raise ValueError("training data checksum does not match the candidate package")

    split = create_split(dataset.labels.to_numpy(), seed=42)
    split_dict = split.to_dict()
    for partition in ("train", "validation", "test"):
        expected = manifest["split"]["partitions"][partition]["indices_sha256"]
        if split_dict["index_sha256"][partition] != expected:
            raise ValueError(f"{partition} split does not match the candidate package")

    X_train = dataset.features.iloc[list(split.train_indices)]
    y_train = dataset.labels.iloc[list(split.train_indices)]
    X_validation = dataset.features.iloc[list(split.validation_indices)]
    y_validation = dataset.labels.iloc[list(split.validation_indices)].to_numpy(dtype=int)
    X_test = dataset.features.iloc[list(split.test_indices)]
    y_test = dataset.labels.iloc[list(split.test_indices)].to_numpy(dtype=int)

    balanced = joblib.load(verified.model_path)
    source_parameters = dict(manifest["model_parameters"])
    unweighted_parameters = controlled_unweighted_parameters(source_parameters)
    unweighted_estimator = build_xgboost_pipeline().set_params(**unweighted_parameters)
    unweighted = calibrate_candidate(unweighted_estimator, X_train, y_train, cv_folds=5)

    balanced_validation_scores = _positive_scores(balanced, X_validation)
    unweighted_validation_scores = _positive_scores(unweighted, X_validation)
    balanced_threshold = select_f2_threshold(y_validation, balanced_validation_scores)
    unweighted_threshold = select_f2_threshold(y_validation, unweighted_validation_scores)
    persisted_threshold = float(source_metrics["threshold"])
    if not np.isclose(
        balanced_threshold.default_threshold,
        persisted_threshold,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("balanced validation threshold does not match candidate evidence")

    balanced_test_scores = _positive_scores(balanced, X_test)
    unweighted_test_scores = _positive_scores(unweighted, X_test)
    balanced_test = evaluate_frozen_scores(
        persisted_threshold, y_test, balanced_test_scores
    ).to_dict()
    for name in ("average_precision", "brier_score", "precision", "recall"):
        if not np.isclose(
            float(balanced_test[name]),
            float(source_metrics[name]),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"recomputed balanced {name} does not match package evidence")

    output = Path(model_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(unweighted, output)
    negative_count = int(np.count_nonzero(y_train.to_numpy() == 0))
    positive_count = int(np.count_nonzero(y_train.to_numpy() == 1))

    return {
        "comparison_type": "posthoc_xgboost_class_weight_ablation",
        "governance_note": (
            "The user explicitly authorized validation and held-out test comparison. "
            "Thresholds were selected only on validation; test results did not affect fitting, "
            "calibration, threshold selection, or active-model state."
        ),
        "protocol": {
            "source_model_version": manifest["version"],
            "data_sha256": dataset.audit.source_sha256,
            "split_seed": 42,
            "train_index_sha256": split_dict["index_sha256"]["train"],
            "validation_index_sha256": split_dict["index_sha256"]["validation"],
            "test_index_sha256": split_dict["index_sha256"]["test"],
            "train_row_count": len(X_train),
            "validation_row_count": len(X_validation),
            "test_row_count": len(X_test),
            "train_positive_count": positive_count,
            "train_negative_count": negative_count,
            "full_training_balance_ratio": negative_count / positive_count,
            "calibration": "5-fold sigmoid on training partition",
            "threshold_selection": "validation F2 maximum; lowest threshold on tie",
            "controlled_change": "classifier__scale_pos_weight: balanced -> 1.0",
            "hyperparameters_reoptimized": False,
            "interpretation": (
                "Controlled ablation of class weighting using the balanced model's frozen "
                "hyperparameters; it is not an independently tuned unweighted model."
            ),
        },
        "balanced": {
            "source": "active candidate model",
            "parameters": source_parameters,
            "resolved_fold_weights": _resolved_weights(balanced),
            "threshold_selection": balanced_threshold.to_dict(),
            "validation_metrics": evaluate_frozen_scores(
                persisted_threshold, y_validation, balanced_validation_scores
            ).to_dict(),
            "test_metrics": balanced_test,
        },
        "unweighted": {
            "source": "controlled refit",
            "parameters": unweighted_parameters,
            "resolved_fold_weights": _resolved_weights(unweighted),
            "threshold_selection": unweighted_threshold.to_dict(),
            "validation_metrics": evaluate_frozen_scores(
                unweighted_threshold.default_threshold,
                y_validation,
                unweighted_validation_scores,
            ).to_dict(),
            "test_metrics": evaluate_frozen_scores(
                unweighted_threshold.default_threshold,
                y_test,
                unweighted_test_scores,
            ).to_dict(),
            "model_file": output.name,
            "model_sha256": _sha256(output),
        },
    }
