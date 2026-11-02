"""Overview and active-model information page."""

from __future__ import annotations

from typing import Any

LIMITATION_TEXT = (
    "資料缺少公司、年度與預測窗口；模型信心分數與分類僅供分析優先順序使用，"
    "不代表因果關係或特定未來事件。"
)


def render_overview(model_info: dict[str, Any], st: Any) -> None:
    """Render only frozen API model metadata; no local model calculation occurs."""
    st.subheader("總覽與模型資訊")
    st.info(LIMITATION_TEXT)
    st.caption("本頁僅使用模型信心分數與分類結果。")
    st.write(f"模型版本：{model_info.get('version', '未知')}")
    st.write(f"模型家族：{model_info.get('model_family', '未知')}")
    st.write(f"建立時間：{model_info.get('created_at', '未知')}")
    columns = st.columns(3)
    columns[0].metric("預設門檻", model_info.get("default_threshold", "—"))
    columns[1].metric("Average Precision", model_info.get("average_precision", "—"))
    columns[2].metric("Recall@Top10%", model_info.get("recall_at_top_10_percent", "—"))
