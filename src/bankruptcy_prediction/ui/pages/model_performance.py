"""Frozen model-performance summary page."""

from __future__ import annotations

from typing import Any

LIMITATIONS = (
    "資料切分：64% 訓練／16% 驗證／20% 最終測試；亂數種子 42。"
    " final test 只評估一次；MVP 門檻為測試 Average Precision 至少為正例比例基準 2 倍，"
    "且 Recall@Top10% 至少 50%。"
)


def render_model_performance(model_info: dict[str, Any], st: Any) -> None:
    st.subheader("模型效能")
    st.info(LIMITATIONS)
    st.write(f"模型版本：{model_info.get('version', '未知')}")
    st.metric("Average Precision", model_info.get("average_precision", "—"))
    st.metric("Recall@Top10%", model_info.get("recall_at_top_10_percent", "—"))
    st.write(f"MVP 門檻結果：{model_info.get('mvp_passed', '未知')}")
    st.caption("PR 曲線、校準曲線與混淆矩陣使用模型包保存的凍結圖表；本頁不即時計算替代圖表。")
