"""Batch upload page; FastAPI remains the validation authority."""

from __future__ import annotations

from typing import Any

from bankruptcy_prediction.ui.api_client import ApiClient, ApiClientError
from bankruptcy_prediction.ui.session_state import save_score_session

UPLOAD_RULES = (
    "上傳限制：CSV 不超過 10 MB、最多 10,000 列，且必須包含正規化後的 95 個特徵。"
    "可選 record_id；未提供時由服務產生 row_000001 格式識別碼。"
)


def render_batch_scoring(
    st: Any, client: ApiClient, *, model_loaded: bool
) -> dict[str, Any] | None:
    st.subheader("批次評分")
    st.info(UPLOAD_RULES)
    if not model_loaded:
        st.warning("沒有 active model，請先啟用模型後再上傳。")
        st.button("送出評分", disabled=True)
        return None
    uploaded = st.file_uploader("選擇 CSV", type=["csv"], disabled=not model_loaded)
    threshold = st.slider("情境門檻", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    if uploaded is None or not st.button("送出評分", disabled=uploaded is None):
        return None
    try:
        payload = client.score(
            uploaded.getvalue(),
            filename=getattr(uploaded, "name", "input.csv"),
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
        uploaded.getvalue(),
        getattr(uploaded, "name", "input.csv"),
    )
    summary = payload.get("summary", {})
    st.write(
        f"總列數：{summary.get('total_rows', 0)}；成功：{summary.get('succeeded_rows', 0)}；"
        f"失敗：{summary.get('failed_rows', 0)}"
    )
    return payload
