"""Build an offline HTML model report with locally bundled evaluation charts."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

NON_CAUSAL_NOTICE = "SHAP 與重要性僅描述模型貢獻，不代表因果關係，亦非改善建議。"
SCOPE_NOTICE = (
    "資料未包含公司識別、年度及預測窗口；模型信心分數不是未來一年破產機率，"
    "結果僅供分析優先順序使用。"
)
DEFAULT_LIMITATIONS = (
    "本模型依既有訓練資料與欄位範圍建立；分數不構成授信、投資或其他自動決策。",
    "未提供的公司、年度及預測窗口不會由系統推測或補造。",
)


@dataclass(frozen=True)
class ModelReportContext:
    dataset_audit: Mapping[str, Any]
    split_summary: Mapping[str, Any]
    search_summary: Mapping[str, Any]
    selection_summary: Mapping[str, Any]
    threshold_summary: Mapping[str, Any]
    metrics_report: Any
    y_test: Any
    confidence_scores: Any
    mean_abs_shap: Mapping[str, float]
    model_family: str
    limitations: tuple[str, ...] = DEFAULT_LIMITATIONS


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _section(title: str, value: Any) -> str:
    encoded = json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
    return f"<section><h2>{html.escape(title)}</h2><pre>{html.escape(encoded)}</pre></section>"


def _plot_pr_curve(y_test: np.ndarray, scores: np.ndarray, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    positives = int(np.sum(y_test == 1))
    if positives:
        precision, recall, _ = precision_recall_curve(y_test, scores)
        axis.plot(recall, precision, color="#1769aa", label="Precision–Recall")
        axis.axhline(
            positives / len(y_test), color="#777777", linestyle="--", label="Positive prevalence"
        )
        axis.legend(loc="best")
    else:
        axis.text(0.5, 0.5, "No positive cases in test data", ha="center", va="center")
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–Recall Curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.2)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def _plot_calibration(y_test: np.ndarray, scores: np.ndarray, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    if np.unique(y_test).size == 2:
        bins = min(10, max(2, len(y_test) // 2))
        observed, predicted = calibration_curve(y_test, scores, n_bins=bins, strategy="quantile")
        axis.plot(predicted, observed, marker="o", color="#00897b", label="Model")
        axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Ideal calibration")
        axis.legend(loc="best")
    else:
        axis.text(0.5, 0.5, "Calibration curve needs both classes", ha="center", va="center")
    axis.set(
        xlabel="Mean model confidence score",
        ylabel="Observed positive fraction",
        title="Calibration Curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.2)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def _plot_confusion_matrix(
    y_test: np.ndarray, scores: np.ndarray, threshold: float, destination: Path
) -> None:
    matrix = np.zeros((2, 2), dtype=int)
    predictions = scores >= threshold
    for actual, predicted in zip(y_test, predictions, strict=True):
        matrix[int(actual), int(predicted)] += 1
    figure, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Not positive", "Positive"],
        yticklabels=["Not positive", "Positive"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"Confusion Matrix (threshold={threshold:.4f})",
    )
    midpoint = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > midpoint else "black",
            )
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def _plot_importance(importance: Mapping[str, float], destination: Path) -> None:
    items = sorted(
        ((str(name), float(value)) for name, value in importance.items()),
        key=lambda item: (-abs(item[1]), item[0]),
    )[:10]
    figure, axis = plt.subplots(figsize=(7.2, max(3.6, 0.35 * len(items))), constrained_layout=True)
    if items:
        names, values = zip(*reversed(items), strict=True)
        axis.barh(names, values, color="#5c6bc0")
        axis.set_xlabel("Mean absolute SHAP contribution")
    else:
        axis.text(0.5, 0.5, "No feature importance supplied", ha="center", va="center")
    axis.set_title("Top Feature Importance")
    axis.grid(axis="x", alpha=0.2)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def build_model_report(context: ModelReportContext, output_dir: str | Path) -> Path:
    """Write a self-contained offline HTML report and four relative chart assets."""
    if not isinstance(context, ModelReportContext):
        raise TypeError("context must be a ModelReportContext")
    labels = np.asarray(context.y_test)
    scores = np.asarray(context.confidence_scores, dtype=float)
    if labels.ndim != 1 or len(labels) == 0 or not set(np.unique(labels).tolist()) <= {0, 1}:
        raise ValueError("y_test must be a non-empty one-dimensional binary array")
    if scores.ndim != 1 or len(scores) != len(labels):
        raise ValueError("confidence_scores must be one-dimensional and match y_test")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("confidence_scores must be finite values between 0 and 1")
    metrics = _jsonable(context.metrics_report)
    if not isinstance(metrics, Mapping) or "threshold" not in metrics:
        raise ValueError("metrics_report must expose a threshold")
    threshold = float(metrics["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("metrics threshold must be between 0 and 1")

    root = Path(output_dir)
    charts = root / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    _plot_pr_curve(labels, scores, charts / "precision_recall.png")
    _plot_calibration(labels, scores, charts / "calibration.png")
    _plot_confusion_matrix(labels, scores, threshold, charts / "confusion_matrix.png")
    _plot_importance(context.mean_abs_shap, charts / "feature_importance.png")

    limitations = tuple(context.limitations) or DEFAULT_LIMITATIONS
    test_only_notice = (
        "<p class=notice>此報告標記為 test-only 測試用途，不可用作正式模型品質或 Release 認證。</p>"
        if context.split_summary.get("test_only") is True
        else ""
    )
    section_html = "\n".join(
        [
            _section("Data audit", context.dataset_audit),
            _section("Train / validation / test split", context.split_summary),
            _section("Hyperparameter search", context.search_summary),
            _section("Model selection", context.selection_summary),
            _section("Default threshold", context.threshold_summary),
            _section("Final test evaluation", metrics),
            "<section><h2>Limitations</h2><ul>"
            + "".join(f"<li>{html.escape(item)}</li>" for item in limitations)
            + "</ul><p class=notice>"
            + html.escape(SCOPE_NOTICE)
            + "</p><p class=notice>"
            + html.escape(NON_CAUSAL_NOTICE)
            + "</p></section>",
        ]
    )
    report = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>企業破產預測系統模型報告</title><style>
body{{font-family:Segoe UI,Microsoft JhengHei,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#202124;line-height:1.55}}
h1,h2{{color:#17365d}}section{{border-top:1px solid #d9e1f2;padding:1rem 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:1rem;border-radius:6px}}
.notice{{background:#fff7e0;border-left:4px solid #edb400;padding:.8rem}}.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}}figure{{margin:0}}img{{width:100%;height:auto}}figcaption{{font-weight:600}}
</style></head><body><h1>企業破產預測系統模型報告</h1>
<p>模型家族：{html.escape(context.model_family)}。本報告使用「模型信心分數」表述，不代表公司未來一年破產機率。</p>
{test_only_notice}
<section><h2>Evaluation charts</h2><div class="charts">
<figure><img src="charts/precision_recall.png" alt="Precision Recall curve"><figcaption>Precision–Recall</figcaption></figure>
<figure><img src="charts/calibration.png" alt="Calibration curve"><figcaption>Calibration</figcaption></figure>
<figure><img src="charts/confusion_matrix.png" alt="Confusion matrix"><figcaption>Confusion matrix</figcaption></figure>
<figure><img src="charts/feature_importance.png" alt="Feature importance"><figcaption>Feature importance (model contribution, not causality)</figcaption></figure>
</div></section>{section_html}</body></html>"""
    destination = root / "model_report.html"
    destination.write_text(report, encoding="utf-8")
    return destination
