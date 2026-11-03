"""Research dashboard regression tests; no model fitting or API access."""

from pathlib import Path

import pytest

from bankruptcy_prediction.ui.pages.research_dashboard import (
    CANDIDATE,
    ResearchDataError,
    _ap_comparison_chart,
    load_class_weight_comparison,
    load_final_test_comparison,
    load_objective_comparison,
    load_research_metrics,
    load_training_comparison,
    render_research_dashboard,
)


class FakeStreamlit:
    def __init__(self):
        self.calls = []

    def columns(self, spec):
        self.calls.append(("columns", (spec,), {}))
        count = spec if isinstance(spec, int) else len(spec)
        return [self] * count

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record


def test_missing_metrics_are_reported_instead_of_fabricated(tmp_path):
    with pytest.raises(ResearchDataError, match="無法讀取研究成果"):
        load_research_metrics(tmp_path)


def test_dashboard_shows_baseline_and_original_charts_when_accessible(tmp_path):
    project = Path(__file__).resolve().parents[2]
    model_dir = tmp_path / "models" / CANDIDATE
    model_dir.mkdir(parents=True)
    for name in ("metrics.json", "model_report.html"):
        (model_dir / name).write_text(
            (project / "models" / CANDIDATE / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    experiment_dir = tmp_path / "experiments" / "xgboost-objectives-20261022"
    experiment_dir.mkdir(parents=True)
    source = project / "experiments" / "xgboost-objectives-20261022" / "comparison.json"
    (experiment_dir / "comparison.json").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    weight_source = project / "experiments" / "xgboost-class-weight-20261103" / "comparison.json"
    weight_dir = tmp_path / "experiments" / "xgboost-class-weight-20261103"
    weight_dir.mkdir(parents=True)
    (weight_dir / "comparison.json").write_text(
        weight_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    posthoc_source = project / "experiments" / "posthoc-final-test-comparison.json"
    (tmp_path / "experiments" / "posthoc-final-test-comparison.json").write_text(
        posthoc_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    comparison_chart_dir = tmp_path / "experiments" / "posthoc-final-test-comparison"
    comparison_chart_dir.mkdir(parents=True)
    (comparison_chart_dir / "precision_recall_comparison.png").write_bytes(b"PNG fixture")
    (comparison_chart_dir / "confusion_matrix_xgboost.png").write_bytes(b"PNG fixture")
    (comparison_chart_dir / "calibration_xgboost_detailed.png").write_bytes(b"PNG fixture")
    (comparison_chart_dir / "shap_feature_importance_top10.png").write_bytes(b"PNG fixture")
    (comparison_chart_dir / "shap_feature_importance_last10.png").write_bytes(b"PNG fixture")
    chart_dir = tmp_path / "models" / CANDIDATE / "charts"
    chart_dir.mkdir(parents=True)
    st = FakeStreamlit()
    render_research_dashboard(st, tmp_path)
    assert len([name for name, _, _ in st.calls if name == "image"]) == 5
    assert any(name == "columns" and args == ([1, 2, 1],) for name, args, _ in st.calls)
    assert not any(name == "metric" for name, _, _ in st.calls)
    charts = [args[0] for name, args, _ in st.calls if name == "altair_chart"]
    assert len(charts) == 2
    assert not any(name == "bar_chart" for name, _, _ in st.calls)
    training_spec = charts[0].to_dict()
    assert training_spec["layer"][0]["mark"]["size"] == 30
    assert training_spec["layer"][0]["encoding"]["color"]["scale"]["range"][1] == "#0B3A75"
    assert load_training_comparison(tmp_path).loc["XGBoost", "Validation AP"] > 0.53
    assert load_objective_comparison(tmp_path).loc["rank:pairwise", "AP"] > 0.57
    weight_comparison = load_class_weight_comparison(tmp_path)
    balanced_test = weight_comparison[
        (weight_comparison["Dataset"] == "Held-out test")
        & (weight_comparison["Class weight"] == "Balanced")
    ].iloc[0]
    unweighted_test = weight_comparison[
        (weight_comparison["Dataset"] == "Held-out test")
        & (weight_comparison["Class weight"] == "Unweighted")
    ].iloc[0]
    assert balanced_test["AP"] > unweighted_test["AP"]
    final_comparison, xgboost, logistic = load_final_test_comparison(tmp_path)
    assert list(final_comparison.columns) == ["指標", "XGBoost binary", "Logistic Regression"]
    assert xgboost["average_precision"] > logistic["average_precision"]


def test_ap_chart_uses_shared_scale_and_thin_deep_blue_xgboost_bar():
    comparison = __import__("pandas").DataFrame(
        {
            "Training CV AP": [0.4075, 0.4448],
            "Validation AP": [0.4235, 0.5366],
        },
        index=["Logistic Regression", "XGBoost"],
    )
    comparison.index.name = "model"
    spec = _ap_comparison_chart(comparison, "Validation AP").to_dict()
    assert spec["height"] == 310
    assert spec["layer"][0]["mark"]["size"] == 30
    assert spec["layer"][0]["encoding"]["y"]["scale"]["domain"] == [0, 0.6]
    color = spec["layer"][0]["encoding"]["color"]["scale"]
    assert color == {
        "domain": ["Logistic Regression", "XGBoost"],
        "range": ["#91A2B8", "#0B3A75"],
    }


def test_load_class_weight_comparison_keeps_validation_and_test_separate(tmp_path):
    output = tmp_path / "experiments" / "xgboost-class-weight-20261103"
    output.mkdir(parents=True)
    metrics = {
        "average_precision": 0.5,
        "precision": 0.3,
        "recall": 0.7,
        "f1": 0.42,
        "brier_score": 0.02,
        "recall_at_top_10_percent": 0.75,
        "false_positive_count": 12,
        "threshold": 0.05,
    }
    (output / "comparison.json").write_text(
        __import__("json").dumps(
            {
                "balanced": {"validation_metrics": metrics, "test_metrics": metrics},
                "unweighted": {"validation_metrics": metrics, "test_metrics": metrics},
            }
        ),
        encoding="utf-8",
    )
    frame = load_class_weight_comparison(tmp_path)
    assert frame[["Dataset", "Class weight"]].values.tolist() == [
        ["Validation", "Balanced"],
        ["Held-out test", "Balanced"],
        ["Validation", "Unweighted"],
        ["Held-out test", "Unweighted"],
    ]


def test_dashboard_handles_permission_denied_without_fabricating_shap(tmp_path, monkeypatch):
    original = Path.is_file

    def guarded_is_file(path):
        if path.suffix == ".png":
            raise PermissionError("access denied")
        return original(path)

    monkeypatch.setattr(Path, "is_file", guarded_is_file)
    st = FakeStreamlit()
    render_research_dashboard(st, tmp_path)
    assert not any(name == "image" for name, _, _ in st.calls)
    assert any(name == "warning" for name, _, _ in st.calls)
    assert not any(name == "bar_chart" for name, _, _ in st.calls)
