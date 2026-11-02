"""Tests for offline report content and relative chart assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from bankruptcy_prediction.reporting.model_report import (
    ModelReportContext,
    build_model_report,
)


def _context() -> ModelReportContext:
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    scores = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.6, 0.4, 0.7])
    return ModelReportContext(
        dataset_audit={"rows": 6819, "features": 95, "high_correlation_threshold": 0.95},
        split_summary={"train": 0.64, "validation": 0.16, "test": 0.20, "seed": 42},
        search_summary={"cv": 5, "n_iter": 30},
        selection_summary={"model_family": "logistic_regression"},
        threshold_summary={"default_threshold": 0.45, "tie_rule": "lowest threshold"},
        metrics_report={
            "average_precision": 0.9,
            "threshold": 0.45,
            "confusion_matrix": confusion_matrix(labels, scores >= 0.45, labels=[0, 1]).tolist(),
        },
        y_test=labels,
        confidence_scores=scores,
        mean_abs_shap={"cash_ratio": 0.25, "debt_ratio": 0.12},
        model_family="logistic_regression",
    )


def test_report_contains_required_sections_notices_and_offline_charts(tmp_path: Path) -> None:
    report_path = build_model_report(_context(), tmp_path)
    html = report_path.read_text(encoding="utf-8")

    for required in (
        "Data audit",
        "Train / validation / test split",
        "Hyperparameter search",
        "Model selection",
        "Default threshold",
        "Final test evaluation",
        "Limitations",
        "模型信心分數",
        "資料未包含公司識別、年度及預測窗口",
        "不代表因果關係",
    ):
        assert required in html
    assert 'src="charts/precision_recall.png"' in html
    assert 'src="charts/calibration.png"' in html
    assert 'src="charts/confusion_matrix.png"' in html
    assert 'src="charts/feature_importance.png"' in html
    for chart in (
        "precision_recall.png",
        "calibration.png",
        "confusion_matrix.png",
        "feature_importance.png",
    ):
        chart_path = tmp_path / "charts" / chart
        assert chart_path.is_file() and chart_path.stat().st_size > 0


def test_report_escapes_untrusted_context_and_is_portable(tmp_path: Path) -> None:
    context = _context()
    malicious = ModelReportContext(
        dataset_audit={"label": "<script>alert(1)</script>"},
        split_summary=context.split_summary,
        search_summary=context.search_summary,
        selection_summary=context.selection_summary,
        threshold_summary=context.threshold_summary,
        metrics_report=context.metrics_report,
        y_test=context.y_test,
        confidence_scores=context.confidence_scores,
        mean_abs_shap=context.mean_abs_shap,
        model_family="logistic_regression",
    )

    report_path = build_model_report(malicious, tmp_path / "model-version")
    html = report_path.read_text(encoding="utf-8")

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert report_path.parent == tmp_path / "model-version"
    assert "../" not in html
