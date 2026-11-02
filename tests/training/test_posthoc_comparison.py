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
