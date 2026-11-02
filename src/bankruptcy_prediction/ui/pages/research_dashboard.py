"""Read-only dashboard backed exclusively by persisted research artifacts."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CANDIDATE = "candidate-20261022T105726Z-155ceb0b"
CHARTS = (
    (
        "Precision–Recall · XGBoost vs Logistic Regression",
        "experiments/posthoc-final-test-comparison/precision_recall_comparison.png",
    ),
    ("Calibration · XGBoost", f"models/{CANDIDATE}/charts/calibration.png"),
    ("Confusion Matrix · XGBoost", f"models/{CANDIDATE}/charts/confusion_matrix.png"),
    ("SHAP Feature Importance · XGBoost", f"models/{CANDIDATE}/charts/feature_importance.png"),
)


class ResearchDataError(RuntimeError):
    """Raised when persisted research evidence is missing or malformed."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchDataError(f"無法讀取研究成果：{path}") from exc
    if not isinstance(value, dict):
        raise ResearchDataError(f"研究成果格式錯誤：{path}")
    return value


def _report_section(root: Path, title: str) -> dict[str, Any]:
    path = root / "models" / CANDIDATE / "model_report.html"
    try:
        document = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ResearchDataError(f"無法讀取模型報告：{path}") from exc
    pattern = rf"<section><h2>{re.escape(title)}</h2><pre>(.*?)</pre></section>"
    match = re.search(pattern, document, flags=re.DOTALL)
    if match is None:
        raise ResearchDataError(f"模型報告缺少區段：{title}")
    try:
        value = json.loads(html.unescape(match.group(1)))
    except json.JSONDecodeError as exc:
        raise ResearchDataError(f"模型報告區段格式錯誤：{title}") from exc
    if not isinstance(value, dict):
        raise ResearchDataError(f"模型報告區段格式錯誤：{title}")
    return value


def load_research_metrics(root: Path = PROJECT_ROOT) -> tuple[dict[str, Any], str]:
    """Load immutable final-test metrics without synthesizing fallback values."""
    path = root / "models" / CANDIDATE / "metrics.json"
    metrics = _read_json(path)
    required = {
        "average_precision",
        "positive_prevalence",
        "ap_to_prevalence_ratio",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "confusion_matrix",
        "false_positive_count",
        "recall_at_top_10_percent",
        "top_10_percent_row_count",
        "top_10_percent_positive_count",
        "sample_count",
        "threshold",
    }
    if not required.issubset(metrics):
        raise ResearchDataError(f"final-test 指標缺少必要欄位：{path}")
    return metrics, str(path.relative_to(root))


def load_training_comparison(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    search = _report_section(root, "Hyperparameter search")
    selection = _report_section(root, "Model selection")
    validation = selection.get("validation_average_precision", {})
    rows = []
    for key, label in (("logistic_regression", "Logistic Regression"), ("xgboost", "XGBoost")):
        search_row = search.get(key, {})
        if (
            not isinstance(search_row, dict)
            or not isinstance(validation, dict)
            or key not in validation
        ):
            raise ResearchDataError("模型報告的訓練比較資料不完整")
        rows.append(
            {
                "model": label,
                "Training CV AP": float(search_row["best_validation_average_precision"]),
                "Validation AP": float(validation[key]),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def _ap_comparison_chart(comparison: pd.DataFrame, metric: str) -> alt.LayerChart:
    """Build a restrained two-model AP chart with a shared, audit-friendly scale."""
    chart_data = (
        comparison[[metric]]
        .reset_index()
        .rename(columns={"model": "Model", metric: "Average Precision"})
    )
    model_order = ["Logistic Regression", "XGBoost"]
    colors = ["#91A2B8", "#0B3A75"]
    base = alt.Chart(chart_data).encode(
        x=alt.X(
            "Model:N",
            sort=model_order,
            axis=alt.Axis(title=None, labelAngle=0, labelPadding=10),
            scale=alt.Scale(paddingInner=0.72, paddingOuter=0.4),
        ),
        y=alt.Y(
            "Average Precision:Q",
            scale=alt.Scale(domain=[0, 0.6]),
            axis=alt.Axis(title="Average Precision", format=".2f", tickCount=7),
        ),
        color=alt.Color(
            "Model:N",
            scale=alt.Scale(domain=model_order, range=colors),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Model:N", title="Model"),
            alt.Tooltip("Average Precision:Q", title=metric, format=".4f"),
        ],
    )
    bars = base.mark_bar(
        size=30,
        cornerRadiusTopLeft=5,
        cornerRadiusTopRight=5,
    )
    labels = base.mark_text(
        dy=-11,
        fontSize=13,
        fontWeight=600,
        color="#AFC4E0",
    ).encode(text=alt.Text("Average Precision:Q", format=".4f"))
    return (
        (bars + labels)
        .properties(height=310)
        .configure_view(strokeOpacity=0)
        .configure_axis(gridColor="#26354A", gridOpacity=0.35, domain=False)
    )


def load_objective_comparison(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    path = root / "experiments" / "xgboost-objectives-20261022" / "comparison.json"
    evidence = _read_json(path)
    rows = []
    for key, label in (
        ("binary_logistic", "binary:logistic"),
        ("rank_pairwise", "rank:pairwise"),
        ("pseudo_huber", "reg:pseudohubererror"),
    ):
        item = evidence.get(key, {})
        metrics = item.get("validation_metrics", {}) if isinstance(item, dict) else {}
        required = ("average_precision", "recall_at_top_10_percent", "ndcg_at_top_10_percent")
        if not all(name in metrics for name in required):
            raise ResearchDataError(f"objective 實驗資料不完整：{key}")
        rows.append(
            {
                "Objective": label,
                "AP": float(metrics["average_precision"]),
                "Recall@Top10%": float(metrics["recall_at_top_10_percent"]),
                "NDCG@Top10%": float(metrics["ndcg_at_top_10_percent"]),
            }
        )
    return pd.DataFrame(rows).set_index("Objective")


def load_final_test_comparison(
    root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load the explicitly authorized post-hoc final-test comparison artifact."""
    path = root / "experiments" / "posthoc-final-test-comparison.json"
    evidence = _read_json(path)
    try:
        xgboost = evidence["xgboost_binary"]["metrics"]
        logistic = evidence["logistic_regression"]["metrics"]
    except (KeyError, TypeError) as exc:
        raise ResearchDataError(f"final-test 比較資料不完整：{path}") from exc
    required = (
        "average_precision",
        "ap_to_prevalence_ratio",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "recall_at_top_10_percent",
        "false_positive_count",
        "threshold",
    )
    if not all(name in xgboost and name in logistic for name in required):
        raise ResearchDataError(f"final-test 比較資料缺少必要指標：{path}")
    rows = (
        (
            "Average Precision",
            f"{float(xgboost['average_precision']):.4f}",
            f"{float(logistic['average_precision']):.4f}",
        ),
        (
            "AP／盛行率倍率",
            f"{float(xgboost['ap_to_prevalence_ratio']):.2f} 倍",
            f"{float(logistic['ap_to_prevalence_ratio']):.2f} 倍",
        ),
        ("Precision", f"{float(xgboost['precision']):.2%}", f"{float(logistic['precision']):.2%}"),
        ("Recall", f"{float(xgboost['recall']):.2%}", f"{float(logistic['recall']):.2%}"),
        ("F1", f"{float(xgboost['f1']):.2%}", f"{float(logistic['f1']):.2%}"),
        (
            "Brier Score",
            f"{float(xgboost['brier_score']):.5f}",
            f"{float(logistic['brier_score']):.5f}",
        ),
        (
            "Recall@Top10%",
            f"{float(xgboost['recall_at_top_10_percent']):.2%}",
            f"{float(logistic['recall_at_top_10_percent']):.2%}",
        ),
        (
            "False positives",
            str(int(xgboost["false_positive_count"])),
            str(int(logistic["false_positive_count"])),
        ),
    )
    frame = pd.DataFrame(rows, columns=("指標", "XGBoost binary", "Logistic Regression"))
    return frame, xgboost, logistic


def _active_version(root: Path) -> str | None:
    try:
        pointer = _read_json(root / "models" / "active.json")
    except ResearchDataError:
        return None
    version = pointer.get("version")
    return version if isinstance(version, str) and version else None


def render_research_dashboard(st: Any, root: Path = PROJECT_ROOT) -> None:
    """Render persisted evidence; missing artifacts remain explicit and localized."""
    st.subheader("Research Dashboard · 模型研究成果")
    active = _active_version(root)
    st.caption(f"歷史候選模型：{CANDIDATE} · Active model：{active or '未啟用'}")
    st.info("下列為既有成果檔的唯讀顯示；不會重新訓練、重測 final test 或推估缺失數值。")

    st.markdown("### XGBoost binary vs Logistic Regression")
    metrics: dict[str, Any] | None = None
    try:
        final_comparison, metrics, logistic_metrics = load_final_test_comparison(root)
        st.dataframe(final_comparison, hide_index=True, width="stretch")
        st.caption(
            "兩模型皆以 training 進行 5-fold sigmoid 校準，並各自使用 validation F2 門檻："
            f"XGBoost {float(metrics['threshold']):.6f}；"
            f"Logistic Regression {float(logistic_metrics['threshold']):.6f}。"
        )
        st.caption(
            "Logistic Regression 為事後授權的 final-test 補評估；資料來源："
            "experiments/posthoc-final-test-comparison.json。"
        )
    except (ResearchDataError, TypeError, ValueError, IndexError, KeyError) as exc:
        st.warning(str(exc))
        try:
            metrics, _ = load_research_metrics(root)
        except ResearchDataError:
            metrics = None

    st.markdown("### Head-to-head · Logistic Regression vs XGBoost")
    try:
        comparison = load_training_comparison(root)
        chart_columns = st.columns(2)
        chart_columns[0].markdown("#### Training CV AP")
        chart_columns[0].altair_chart(
            _ap_comparison_chart(comparison, "Training CV AP"),
            width="stretch",
        )
        chart_columns[1].markdown("#### Validation AP")
        chart_columns[1].altair_chart(
            _ap_comparison_chart(comparison, "Validation AP"),
            width="stretch",
        )
        st.dataframe(comparison.style.format("{:.4f}"), width="stretch")
        improvement = (
            comparison.loc["XGBoost", "Validation AP"]
            / comparison.loc["Logistic Regression", "Validation AP"]
            - 1
        ) * 100
        st.caption(
            f"XGBoost validation AP 相對提升 {improvement:.1f}%。兩模型均採 30 組搜尋 × 5-fold；"
            "原始選模階段未以 final test 比較模型；上方 Logistic Regression 數值為事後明確授權的補評估。"
        )
    except (ResearchDataError, TypeError, ValueError, KeyError, ZeroDivisionError) as exc:
        st.warning(str(exc))

    st.markdown("### Validation objective experiments · XGBoost")
    try:
        objectives = load_objective_comparison(root)
        st.dataframe(objectives.style.format("{:.4f}"), width="stretch")
        st.caption(
            "資料來源：experiments/xgboost-objectives-20261022/comparison.json；這是 validation 探索實驗，不是 final-test 競賽。"
        )
    except (ResearchDataError, TypeError, ValueError) as exc:
        st.warning(str(exc))

    st.markdown("### Evaluation charts & SHAP explanations")
    for index in range(0, len(CHARTS), 2):
        cols = st.columns(2)
        for col, (name, filename) in zip(cols, CHARTS[index : index + 2], strict=True):
            col.markdown(f"**{name}**")
            image_path = root / filename
            try:
                if image_path.is_file():
                    col.image(str(image_path), width="stretch")
                elif filename.endswith("confusion_matrix.png") and metrics is not None:
                    matrix = metrics["confusion_matrix"]
                    col.dataframe(
                        pd.DataFrame(
                            matrix,
                            index=["Actual negative", "Actual positive"],
                            columns=["Predicted negative", "Predicted positive"],
                        ),
                        width="stretch",
                    )
                else:
                    col.warning(f"成果圖檔不存在：{filename}")
            except OSError:
                col.warning(f"成果圖檔無法讀取：{filename}")
    st.caption(
        "PR curve 為事後授權比較：兩模型使用同一份 held-out final test（n=1,364；正例=44）；"
        "AP 為曲線摘要，虛線為正例盛行率基準。其餘三圖仍屬 XGBoost active candidate。"
    )
    st.caption("模型信心分數僅供分析優先順序使用；SHAP 是模型貢獻，不代表因果關係。")
