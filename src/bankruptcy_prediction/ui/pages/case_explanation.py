"""Per-record explanation page backed by the current in-memory scoring batch."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd

EXPLANATION_LIMIT = 5


def render_case_explanation(
    st: Any, payload: dict[str, Any], csv_bytes: bytes | None = None
) -> None:
    st.subheader("個案解釋")
    rows = [row for row in payload.get("rows", []) if row.get("row_status") == "success"]
    if not rows:
        st.info("目前批次沒有可顯示的成功列。")
        return
    ids = [str(row["record_id"]) for row in rows]
    selected = st.selectbox("選擇 record_id", ids)
    row = next(item for item in rows if str(item["record_id"]) == selected)
    st.write(f"模型版本：{row.get('model_version', '未知')}")
    st.write(f"模型信心分數：{row.get('confidence_score', '—')}")
    st.write(f"分類：{row.get('risk_class', '—')}")
    st.caption("以下貢獻值描述模型計算的關聯，不代表因果關係。")
    if csv_bytes is not None:
        raw = pd.read_csv(BytesIO(csv_bytes))
        if "record_id" in raw.columns:
            selected_values = raw[raw["record_id"].astype(str) == selected]
        else:
            position = ids.index(selected)
            selected_values = raw.iloc[[position]] if position < len(raw) else raw.iloc[0:0]
        st.write("原始輸入值")
        st.dataframe(selected_values)
    st.write("正向模型貢獻（最多五項）")
    st.dataframe(pd.DataFrame(row.get("positive_factors", [])[:EXPLANATION_LIMIT]))
    st.write("負向模型貢獻（最多五項）")
    st.dataframe(pd.DataFrame(row.get("negative_factors", [])[:EXPLANATION_LIMIT]))
