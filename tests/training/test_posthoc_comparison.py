from pathlib import Path

import numpy as np
import pytest

from bankruptcy_prediction.training import posthoc_comparison


def test_scaler_accepts_only_persisted_search_options():
    assert type(posthoc_comparison._scaler("StandardScaler()")).__name__ == "StandardScaler"
    assert type(posthoc_comparison._scaler("RobustScaler()")).__name__ == "RobustScaler"
    with pytest.raises(ValueError, match="unsupported persisted scaler"):
        posthoc_comparison._scaler("MinMaxScaler()")


def test_report_section_reads_structured_json(tmp_path: Path):
    report = tmp_path / "report.html"
    report.write_text(
        "<section><h2>Model selection</h2><pre>{&quot;winner&quot;: &quot;xgboost&quot;}</pre></section>",
        encoding="utf-8",
    )
    assert posthoc_comparison._report_section(report, "Model selection") == {"winner": "xgboost"}


def test_precision_recall_comparison_chart_is_written(tmp_path: Path):
    output = posthoc_comparison.plot_precision_recall_comparison(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.9, 0.2, 0.8]),
        np.array([0.2, 0.7, 0.3, 0.6]),
        1.0,
        1.0,
        tmp_path / "comparison.png",
    )
    assert output.read_bytes().startswith(b"\x89PNG")


def test_labeled_confusion_matrix_chart_is_written(tmp_path: Path):
    output = posthoc_comparison.plot_labeled_confusion_matrix(
        [[1262, 58], [15, 29]],
        0.049794,
        tmp_path / "confusion.png",
    )
    assert output.read_bytes().startswith(b"\x89PNG")


def test_labeled_confusion_matrix_rejects_invalid_shape(tmp_path: Path):
    with pytest.raises(ValueError, match="2x2"):
        posthoc_comparison.plot_labeled_confusion_matrix(
            [[1, 2, 3]], 0.5, tmp_path / "confusion.png"
        )


def test_detailed_calibration_writes_chart_and_preserves_bin_totals(tmp_path: Path):
    labels = np.array([0] * 16 + [1] * 4)
    scores = np.linspace(0.01, 0.40, 20)
    output, evidence = posthoc_comparison.plot_detailed_calibration(
        labels,
        scores,
        tmp_path / "calibration.png",
        n_bins=5,
    )
    assert output.read_bytes().startswith(b"\x89PNG")
    assert sum(evidence["sample_count"]) == 20
    assert sum(evidence["positive_count"]) == 4
    assert len(evidence["mean_score"]) == 5


def test_calibration_bin_summary_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        posthoc_comparison._calibration_bin_summary(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.2, np.nan, 0.4]),
            n_bins=2,
        )


def test_shap_extreme_charts_use_disjoint_rankings(tmp_path: Path):
    importance = {f"feature_{index:02d}": float(index) for index in range(25)}
    top_path, last_path, top, last = posthoc_comparison.plot_shap_importance_extremes(
        importance,
        tmp_path / "top.png",
        tmp_path / "last.png",
    )
    assert top_path.read_bytes().startswith(b"\x89PNG")
    assert last_path.read_bytes().startswith(b"\x89PNG")
    assert top[0] == ("feature_24", 24.0)
    assert last[0] == ("feature_00", 0.0)
    assert {name for name, _ in top}.isdisjoint(name for name, _ in last)
