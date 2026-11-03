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
    (
        "Calibration · XGBoost",
        "experiments/posthoc-final-test-comparison/calibration_xgboost_detailed.png",
    ),
    (
        "Confusion Matrix · XGBoost",
        "experiments/posthoc-final-test-comparison/confusion_matrix_xgboost.png",
    ),
    (
        "SHAP Feature Importance · Top 10",
        "experiments/posthoc-final-test-comparison/shap_feature_importance_top10.png",
    ),
    (
        "SHAP Feature Importance · Last 10",
        "experiments/posthoc-final-test-comparison/shap_feature_importance_last10.png",
    ),
)
CHART_ROWS = (CHARTS[:2], CHARTS[2:3], CHARTS[3:])

# Rank order comes from the frozen candidate's mean-absolute-SHAP chart.
# This is a bilingual glossary, not a claim about SHAP sign or causality.
SHAP_TOP_FEATURES = (
    (1, "Quick Ratio", "速動比率", "不靠出售存貨，手邊較容易變現的資產夠不夠償還短期債務？"),
    (
        2,
        "Total debt/Total net worth",
        "總負債／股東權益比",
        "公司背的債相對於股東自己的本錢有多少？",
    ),
    (
        3,
        "Persistent EPS in the Last Four Seasons",
        "近四季每股盈餘",
        "最近四季平均每一股能賺多少，幫助觀察企業的獲利表現。",
    ),
    (
        4,
        "Non-industry income and expenditure/revenue",
        "業外收支占營收比重",
        "投資、匯兌或處分資產等本業以外的淨收支，和營業收入相比占多少？",
    ),
    (
        5,
        "Allocation rate per person",
        "每位員工分攤固定資產",
        "每位員工平均對應多少機器、設備或廠房等固定資產；不是薪資分配率。",
    ),
    (
        6,
        "ROA(C) before interest and depreciation before interest",
        "資產報酬率（C，息折前口徑）",
        "公司的資產產生獲利的能力如何？這是排除部分利息、折舊影響的資產報酬指標。",
    ),
    (
        7,
        "Interest-bearing debt interest rate",
        "有息負債相關比率",
        "與需要付利息的借款有關；原資料欄位對照為「有息負債／股東權益」，不能直接當銀行貸款利率。",
    ),
    (
        8,
        "Borrowing dependency",
        "借款依賴／有息負債成本",
        "觀察借款融資負擔；原資料欄位對照為「有息負債成本」，不宜只按名稱解讀成借款占比。",
    ),
    (
        9,
        "Continuous interest rate (after tax)",
        "經常性稅後淨利率",
        "排除處分資產等一次性損益後，企業每一元營收大約留下多少淨利？這不是存款利率。",
    ),
    (
        10,
        "Continuous Net Profit Growth Rate",
        "經常性淨利成長率",
        "不靠賣資產等一次性收益，公司的淨利是在成長還是衰退？",
    ),
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


def load_class_weight_comparison(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Load the controlled balanced-versus-unweighted XGBoost experiment."""
    path = root / "experiments" / "xgboost-class-weight-20261103" / "comparison.json"
    evidence = _read_json(path)
    rows = []
    for model_key, model_label in (("balanced", "Balanced"), ("unweighted", "Unweighted")):
        item = evidence.get(model_key)
        if not isinstance(item, dict):
            raise ResearchDataError(f"class-weight 實驗資料不完整：{model_key}")
        for partition_key, partition_label in (
            ("validation_metrics", "Validation"),
            ("test_metrics", "Held-out test"),
        ):
            metrics = item.get(partition_key)
            required = (
                "average_precision",
                "precision",
                "recall",
                "f1",
                "brier_score",
                "recall_at_top_10_percent",
                "false_positive_count",
                "threshold",
            )
            if not isinstance(metrics, dict) or not all(name in metrics for name in required):
                raise ResearchDataError(f"class-weight 實驗指標不完整：{model_key}.{partition_key}")
            rows.append(
                {
                    "Dataset": partition_label,
                    "Class weight": model_label,
                    "AP": float(metrics["average_precision"]),
                    "Precision": float(metrics["precision"]),
                    "Recall": float(metrics["recall"]),
                    "F1": float(metrics["f1"]),
                    "Brier": float(metrics["brier_score"]),
                    "Recall@Top10%": float(metrics["recall_at_top_10_percent"]),
                    "False positives": int(metrics["false_positive_count"]),
                    "Threshold": float(metrics["threshold"]),
                }
            )
    return pd.DataFrame(rows)


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

    st.markdown("### XGBoost class weight · Balanced vs Unweighted")
    try:
        class_weight = load_class_weight_comparison(root)
        st.dataframe(
            class_weight.style.format(
                {
                    "AP": "{:.4f}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1": "{:.2%}",
                    "Brier": "{:.5f}",
                    "Recall@Top10%": "{:.2%}",
                    "Threshold": "{:.6f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "唯一控制變因為 scale_pos_weight：Balanced 由每個 training fold 的負／正樣本比估計，"
            "Unweighted 固定為 1.0；其他 XGBoost 超參數及切分相同。兩者門檻均只以 validation F2 選定。"
        )
        st.caption(
            "Held-out test 比較為使用者明確授權的事後消融實驗，不回饋訓練、門檻或 active model；"
            "Unweighted 未獨立重做超參數搜尋。"
        )
    except (ResearchDataError, TypeError, ValueError, KeyError) as exc:
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
    for chart_row in CHART_ROWS:
        if len(chart_row) == 1:
            # A centered half-width column keeps a standalone square matrix at the
            # same visual scale as either chart in the surrounding two-column rows.
            cols = (st.columns([1, 2, 1])[1],)
        else:
            cols = st.columns(len(chart_row))
        for col, (name, filename) in zip(cols, chart_row, strict=True):
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
        "AP 為曲線摘要，虛線為正例盛行率基準；其餘圖表仍屬 XGBoost active candidate。"
    )
    st.caption(
        "SHAP Top 10／Last 10 使用相同 validation split，對五個校準基模型的完整特徵 SHAP 取平均絕對值；"
        "Last 10 代表本資料與模型下平均貢獻最低，不代表因果無關或永遠不重要。"
    )
    st.markdown("#### SHAP Top 10｜中英文白話解讀")
    st.caption("排名依目前 XGBoost 候選模型的平均絕對 SHAP 貢獻，由高至低。")
    st.dataframe(
        pd.DataFrame(
            SHAP_TOP_FEATURES,
            columns=["重要性排名", "English feature", "中文名稱", "白話解釋"],
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "長條越長，代表平均影響力越大；但這張圖不顯示影響方向。"
        "不能由此判斷某指標提高一定會讓風險上升或下降。"
        "少數原始英文欄名與資料字典的實際定義不同，已在中文說明中註明。"
    )
    st.caption("模型信心分數僅供分析優先順序使用；SHAP 是模型貢獻，不代表因果關係。")
