"""Batch upload page; FastAPI remains the validation authority."""

from __future__ import annotations

from typing import Any

from bankruptcy_prediction.ui.api_client import ApiClient, ApiClientError
from bankruptcy_prediction.ui.session_state import (
    clear_score_session,
    get_score_session,
    save_score_session,
)

UPLOAD_RULES = (
    "上傳限制：CSV 不超過 10 MB、最多 10,000 列，且必須包含正規化後的 95 個特徵。"
    "可選 record_id；未提供時由服務產生 row_000001 格式識別碼。"
)


def render_batch_scoring(
    st: Any,
    client: ApiClient,
    *,
    model_loaded: bool,
    default_threshold: float = 0.5,
) -> dict[str, Any] | None:
    st.subheader("批次評分")
    st.info(UPLOAD_RULES)
    if not model_loaded:
        st.warning("沒有 active model，請先啟用模型後再上傳。")
        st.button("送出評分", disabled=True)
        return None
    uploaded = st.file_uploader("選擇 CSV", type=["csv"], disabled=not model_loaded)
    threshold = st.slider(
        "初始情境門檻",
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.000001,
        format="%.6f",
        key="upload_threshold",
    )
    if uploaded is None:
        return None
    uploaded_bytes = uploaded.getvalue()
    uploaded_name = getattr(uploaded, "name", "input.csv")
    previous = get_score_session(st.session_state)
    if previous is not None and (previous[1] != uploaded_bytes or previous[2] != uploaded_name):
        clear_score_session(st.session_state)
        st.info("已選擇新的檔案；送出評分前不顯示上一批結果。")
    if not st.button("送出評分", disabled=False):
        return None
    try:
        payload = client.score(
            uploaded_bytes,
            filename=uploaded_name,
            threshold=threshold,
        )
    except ApiClientError as exc:
        st.error(str(exc))
        if exc.status_code in {422, 413, 503}:
            st.info("請依錯誤提示修正檔案、大小／列數或模型狀態後重試。")
        return None
    save_score_session(
        st.session_state,
        payload,
        uploaded_bytes,
        uploaded_name,
    )
    summary = payload.get("summary", {})
    st.write(
        f"總列數：{summary.get('total_rows', 0)}；成功：{summary.get('succeeded_rows', 0)}；"
        f"失敗：{summary.get('failed_rows', 0)}"
    )
    return payload
