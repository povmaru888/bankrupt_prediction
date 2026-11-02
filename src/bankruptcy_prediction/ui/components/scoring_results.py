"""Batch scoring result display and ephemeral threshold scenarios."""

from __future__ import annotations

from typing import Any

import pandas as pd

from bankruptcy_prediction.ui.api_client import ApiClient, ApiClientError


def _rows_frame(payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(payload.get("rows", []))


def render_scoring_results(
    st: Any,
    client: ApiClient,
    payload: dict[str, Any],
    *,
    csv_bytes: bytes | None = None,
    filename: str = "input.csv",
) -> dict[str, Any]:
    """Render results and request a new scenario when the threshold changes."""
    st.subheader("分數分布與高風險名單")
    current = payload
    original_threshold = float(payload.get("summary", {}).get("threshold_used", 0.5))
    threshold = st.slider(
        "情境門檻",
        min_value=0.0,
        max_value=1.0,
        value=original_threshold,
        step=0.000001,
        format="%.6f",
        key="result_threshold",
    )
    if threshold != original_threshold and csv_bytes is not None:
        try:
            current = client.score(csv_bytes, filename=filename, threshold=threshold)
            st.session_state["score_response"] = current
        except ApiClientError as exc:
            st.error(str(exc))
            return payload

    frame = _rows_frame(current)
    successful = frame[frame.get("row_status", pd.Series(dtype=str)) == "success"]
    high_risk = (
        successful[successful["risk_class"] == "high_risk"] if not successful.empty else successful
    )
    top_ten = successful[successful["top_10_percent_flag"]] if not successful.empty else successful
    st.write(f"成功列：{len(successful)}；高風險列：{len(high_risk)}")
    st.caption(f"目前結果使用門檻：{float(current.get('summary', {}).get('threshold_used', threshold)):.6f}")
    if not successful.empty:
        st.bar_chart(successful["confidence_score"])
    st.dataframe(high_risk)
    st.caption("Top 10% 是批次內排序結果，不受情境門檻調整影響。")
    st.dataframe(top_ten)
    if csv_bytes is not None and st.button("下載結果 CSV"):
        try:
            csv_result = client.score(
                csv_bytes, filename=filename, threshold=threshold, response_format="csv"
            )
            st.download_button("下載", csv_result, file_name="scoring_results.csv", mime="text/csv")
        except ApiClientError as exc:
            st.error(str(exc))
    return current
