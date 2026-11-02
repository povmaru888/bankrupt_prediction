"""Explicitly authorized post-hoc final-test comparison for the baseline model."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import RobustScaler, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bankruptcy_prediction.data.feature_schema import build_feature_schema
from bankruptcy_prediction.data.training_loader import load_training_data
from bankruptcy_prediction.evaluation.metrics import evaluate_frozen_scores
from bankruptcy_prediction.model_package.verifier import verify_package
from bankruptcy_prediction.training.calibration import calibrate_candidate
from bankruptcy_prediction.training.logistic_pipeline import build_logistic_pipeline
from bankruptcy_prediction.training.split_manager import create_split
from bankruptcy_prediction.training.threshold_selector import select_f2_threshold


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _report_section(report_path: Path, title: str) -> dict[str, Any]:
    document = report_path.read_text(encoding="utf-8")
    match = re.search(
        rf"<section><h2>{re.escape(title)}</h2><pre>(.*?)</pre></section>",
        document,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"model report is missing section: {title}")
    value = json.loads(html.unescape(match.group(1)))
    if not isinstance(value, dict):
        raise ValueError(f"model report section must be an object: {title}")
    return value


def _scaler(value: Any) -> Any:
    if value == "StandardScaler()":
        return StandardScaler()
    if value == "RobustScaler()":
        return RobustScaler()
    raise ValueError(f"unsupported persisted scaler: {value!r}")


def _positive_scores(model: Any, features: Any) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("model predict_proba must return two columns")
    return probabilities[:, 1]


def plot_precision_recall_comparison(
    labels: np.ndarray,
    xgboost_scores: np.ndarray,
    logistic_scores: np.ndarray,
    xgboost_ap: float,
    logistic_ap: float,
    output_path: str | Path,
) -> Path:
    """Plot both models on the same held-out observations and prevalence baseline."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    xgb_precision, xgb_recall, _ = precision_recall_curve(labels, xgboost_scores)
    lr_precision, lr_recall, _ = precision_recall_curve(labels, logistic_scores)
    prevalence = float(np.mean(labels))

    fig, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    axis.step(
        xgb_recall,
        xgb_precision,
        where="post",
        color="#0B3A75",
        linewidth=2.3,
        label=f"XGBoost binary (AP = {xgboost_ap:.4f})",
    )
    axis.step(
        lr_recall,
        lr_precision,
        where="post",
        color="#B46A3C",
        linewidth=2.1,
        linestyle="--",
        label=f"Logistic Regression (AP = {logistic_ap:.4f})",
    )
    axis.axhline(
        prevalence,
        color="#6B7280",
        linewidth=1.4,
        linestyle=":",
        label=f"Positive prevalence = {prevalence:.4f}",
    )
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–Recall Curve · Held-out Final Test",
    )
    axis.grid(True, color="#D8DEE8", linewidth=0.7, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", frameon=False, fontsize=10)
    axis.text(
        0.0,
        -0.16,
        f"Same stratified test partition: n = {len(labels):,}, positives = {int(np.sum(labels)):,}",
        transform=axis.transAxes,
        fontsize=9,
        color="#4B5563",
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return output


def plot_labeled_confusion_matrix(
    matrix: Any,
    threshold: float,
    output_path: str | Path,
) -> Path:
    """Render an equal-area, explicitly labeled binary confusion matrix."""
    values = np.asarray(matrix, dtype=int)
    if values.shape != (2, 2) or np.any(values < 0):
        raise ValueError("confusion matrix must be a non-negative 2x2 matrix")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8.6, 5.8))
    fig.subplots_adjust(left=0.12, right=0.74, top=0.80, bottom=0.18)
    labels = (("TN", "FP"), ("FN", "TP"))
    fills = (("#E8F1FC", "#FFF0E5"), ("#FFF0E5", "#E8F1FC"))
    accents = (("#0B3A75", "#A14A22"), ("#A14A22", "#0B3A75"))
    for row in range(2):
        for column in range(2):
            axis.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=fills[row][column],
                    edgecolor="#1F2937",
                    linewidth=2.5,
                )
            )
            axis.text(
                column,
                row - 0.08,
                labels[row][column],
                ha="center",
                va="center",
                fontsize=19,
                fontweight="bold",
                color=accents[row][column],
            )
            axis.text(
                column,
                row + 0.20,
                f"{values[row, column]:,}",
                ha="center",
                va="center",
                fontsize=16,
                color="#111827",
            )

    axis.set_xlim(-0.5, 1.5)
    axis.set_ylim(1.5, -0.5)
    axis.set_aspect("equal")
    axis.set_xticks((0, 1), labels=("0", "1"), fontsize=15)
    axis.set_yticks((0, 1), labels=("0", "1"), fontsize=15)
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("bottom")
    axis.tick_params(length=0, pad=9)
    axis.set_xlabel("Predicted label", fontsize=15, labelpad=22)
    axis.set_ylabel("Actual label", fontsize=15, labelpad=12)
    for spine in axis.spines.values():
        spine.set_visible(False)

    fig.suptitle("Confusion Matrix · XGBoost", fontsize=20, fontweight="semibold", y=0.95)
    axis.set_title(
        f"Held-out final test · threshold = {threshold:.4f}",
        fontsize=11,
        color="#4B5563",
        pad=34,
    )
    fig.text(0.78, 0.59, "0 : Negative", fontsize=15, color="#111827")
    fig.text(0.78, 0.51, "1 : Positive", fontsize=15, color="#111827")
    fig.text(0.78, 0.38, "TN  True negative", fontsize=10, color="#0B3A75")
    fig.text(0.78, 0.33, "FP  False positive", fontsize=10, color="#A14A22")
    fig.text(0.78, 0.28, "FN  False negative", fontsize=10, color="#A14A22")
    fig.text(0.78, 0.23, "TP  True positive", fontsize=10, color="#0B3A75")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return output


def evaluate_posthoc_logistic(
    data_path: str | Path,
    candidate_dir: str | Path,
    chart_output: str | Path | None = None,
    confusion_chart_output: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the persisted LR winner and evaluate the original final-test partition once."""
    candidate = Path(candidate_dir)
    verified = verify_package(candidate)
    manifest = _read_json(candidate / "manifest.json")
    xgboost_metrics = _read_json(candidate / "metrics.json")
    search = _report_section(candidate / "model_report.html", "Hyperparameter search")
    selection = _report_section(candidate / "model_report.html", "Model selection")

    dataset = load_training_data(data_path, test_only=False)
    schema = build_feature_schema(dataset)
    dataset.features.columns = [field.name for field in schema.features]
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
    y_validation = dataset.labels.iloc[list(split.validation_indices)]
    X_test = dataset.features.iloc[list(split.test_indices)]
    y_test = dataset.labels.iloc[list(split.test_indices)]

    persisted = search["logistic_regression"]
    parameters = dict(persisted["best_parameters"])
    parameters["scaler"] = _scaler(parameters["scaler"])
    estimator = build_logistic_pipeline().set_params(**parameters)
    calibrated = calibrate_candidate(estimator, X_train, y_train, cv_folds=5)

    validation_scores = _positive_scores(calibrated, X_validation)
    threshold = select_f2_threshold(y_validation, validation_scores)
    test_scores = _positive_scores(calibrated, X_test)
    logistic_metrics = evaluate_frozen_scores(
        threshold.default_threshold,
        y_test.to_numpy(),
        test_scores,
    ).to_dict()

    xgboost_model = joblib.load(verified.model_path)
    xgboost_test_scores = _positive_scores(xgboost_model, X_test)
    xgboost_recomputed = evaluate_frozen_scores(
        float(xgboost_metrics["threshold"]),
        y_test.to_numpy(),
        xgboost_test_scores,
    ).to_dict()
    for metric_name in ("average_precision", "brier_score", "precision", "recall"):
        if not np.isclose(
            float(xgboost_recomputed[metric_name]),
            float(xgboost_metrics[metric_name]),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"recomputed XGBoost {metric_name} does not match package evidence")

    chart_evidence: dict[str, Any] = {
        "partition": "held_out_final_test",
        "posthoc_comparison": True,
        "sample_count": len(X_test),
        "positive_count": int(y_test.sum()),
        "positive_prevalence": float(y_test.mean()),
        "xgboost_average_precision": float(xgboost_recomputed["average_precision"]),
        "logistic_average_precision": float(logistic_metrics["average_precision"]),
    }
    if chart_output is not None:
        chart_path = plot_precision_recall_comparison(
            y_test.to_numpy(),
            xgboost_test_scores,
            test_scores,
            float(xgboost_recomputed["average_precision"]),
            float(logistic_metrics["average_precision"]),
            chart_output,
        )
        chart_evidence["chart_sha256"] = hashlib.sha256(chart_path.read_bytes()).hexdigest()

    confusion_evidence: dict[str, Any] = {
        "partition": "held_out_final_test",
        "threshold": float(xgboost_metrics["threshold"]),
        "matrix": xgboost_metrics["confusion_matrix"],
    }
    if confusion_chart_output is not None:
        confusion_chart_path = plot_labeled_confusion_matrix(
            xgboost_metrics["confusion_matrix"],
            float(xgboost_metrics["threshold"]),
            confusion_chart_output,
        )
        confusion_evidence["chart_sha256"] = hashlib.sha256(
            confusion_chart_path.read_bytes()
        ).hexdigest()

    serializable_parameters = dict(persisted["best_parameters"])
    return {
        "comparison_type": "posthoc_final_test_comparison",
        "governance_note": (
            "Logistic Regression final-test evaluation was explicitly authorized after the "
            "original XGBoost-only final evaluation; no model or threshold was selected on test data."
        ),
        "protocol": {
            "data_sha256": dataset.audit.source_sha256,
            "split_seed": 42,
            "search_candidates": int(persisted["attempted_candidates"]),
            "search_cv_folds": 5,
            "calibration": "5-fold sigmoid on training partition",
            "threshold_selection": "validation F2 maximum; lowest threshold on tie",
            "test_partition_sha256": split_dict["index_sha256"]["test"],
            "test_row_count": len(X_test),
        },
        "precision_recall_comparison": chart_evidence,
        "xgboost_confusion_matrix_chart": confusion_evidence,
        "xgboost_binary": {
            "model_version": manifest["version"],
            "metrics": xgboost_metrics,
        },
        "logistic_regression": {
            "training_cv_average_precision": float(persisted["best_validation_average_precision"]),
            "validation_average_precision": float(
                selection["validation_average_precision"]["logistic_regression"]
            ),
            "best_parameters": serializable_parameters,
            "threshold_selection": threshold.to_dict(),
            "metrics": logistic_metrics,
        },
    }
