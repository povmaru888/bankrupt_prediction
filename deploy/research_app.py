"""Lightweight, snapshot-only Streamlit research results page."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
SNAPSHOT_DIR = HERE / "research_snapshot"
SNAPSHOT_PATH = SNAPSHOT_DIR / "snapshot.json"


@st.cache_data(show_spinner=False)
def load_snapshot() -> dict:
    try:
        payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("研究成果快照遺失或損毀，請先重新匯出。") from exc
    required = {
        "schema_version",
        "candidate",
        "final_test_comparison",
        "training_comparison",
        "class_weight_comparison",
        "objective_comparison",
        "images",
        "shap_top_features",
    }
    if payload.get("schema_version") != 1 or not required.issubset(payload):
        raise RuntimeError("研究成果快照版本或欄位不相容。")
    for image in payload["images"]:
        path = SNAPSHOT_DIR / image["file"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != image["sha256"]:
            raise RuntimeError(f"成果圖檔遺失或校驗失敗：{image['file']}")
    return payload


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> None:
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = [str(row.get(key, "—")).replace("|", "\\|") for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    st.markdown("\n".join(lines))


def main() -> None:
    st.set_page_config(page_title="企業破產預測研究成果", page_icon="📊", layout="wide")
    st.title("企業破產預測 · 研究成果")
    st.caption("唯讀成果快照｜不訓練、不推論、不啟動 API")

    try:
        data = load_snapshot()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.info(
        f"候選模型：{data['candidate']}｜Active model：{data.get('active_model') or '未啟用'}｜"
        f"快照時間：{data.get('generated_at_utc', '未知')}"
    )

    st.header("XGBoost binary vs Logistic Regression")
    markdown_table(
        data["final_test_comparison"],
        [("指標", "指標"), ("XGBoost binary", "XGBoost"), ("Logistic Regression", "Logistic Regression")],
    )
    thresholds = data["final_test_thresholds"]
    st.caption(
        f"Validation F2 門檻：XGBoost {thresholds['xgboost']:.6f}；"
        f"Logistic Regression {thresholds['logistic_regression']:.6f}。"
    )

    st.header("模型研究比較")
    left, right = st.columns(2)
    with left:
        st.subheader("Training / Validation AP")
        markdown_table(
            data["training_comparison"],
            [("model", "模型"), ("Training CV AP", "Training CV AP"), ("Validation AP", "Validation AP")],
        )
    with right:
        st.subheader("Validation objectives")
        markdown_table(
            data["objective_comparison"],
            [("Objective", "Objective"), ("AP", "AP"), ("Recall@Top10%", "Recall@Top10%"), ("NDCG@Top10%", "NDCG@Top10%")],
        )

    st.subheader("Class weight · Balanced vs Unweighted")
    markdown_table(
        data["class_weight_comparison"],
        [("Dataset", "Dataset"), ("Class weight", "Class weight"), ("AP", "AP"), ("Recall", "Recall"), ("F1", "F1"), ("Recall@Top10%", "Recall@Top10%")],
    )

    st.header("評估圖表與 SHAP")
    images = data["images"]
    if len(images) == 5:
        # Row 1: PR curve & Calibration (2 charts)
        # Row 2: Confusion Matrix (1 square chart, centered at half width)
        # Row 3: SHAP Top 10 & SHAP Last 10 (2 charts)
        chart_rows = [images[:2], images[2:3], images[3:]]
    else:
        chart_rows = [images[start : start + 2] for start in range(0, len(images), 2)]

    for row in chart_rows:
        if len(row) == 1:
            cols = (st.columns([1, 2, 1])[1],)
        else:
            cols = st.columns(len(row))
        for column, image in zip(cols, row, strict=True):
            with column:
                st.subheader(image["title"])
                st.image(str(SNAPSHOT_DIR / image["file"]), width="stretch")

    st.subheader("SHAP Top 10｜中英文白話解讀")
    markdown_table(
        data["shap_top_features"],
        [("rank", "排名"), ("english", "English feature"), ("chinese", "中文名稱"), ("explanation", "白話解釋")],
    )
    st.caption("模型信心分數僅供分析優先順序使用；SHAP 是模型貢獻，不代表因果關係。")


if __name__ == "__main__":
    main()
