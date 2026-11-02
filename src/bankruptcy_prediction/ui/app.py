"""Streamlit application shell with four API-backed pages."""

from __future__ import annotations

from typing import Any

from bankruptcy_prediction.ui.api_client import ApiClient, ApiClientError
from bankruptcy_prediction.ui.components.scoring_results import render_scoring_results
from bankruptcy_prediction.ui.pages.batch_scoring import render_batch_scoring
from bankruptcy_prediction.ui.pages.case_explanation import render_case_explanation
from bankruptcy_prediction.ui.pages.model_performance import render_model_performance
from bankruptcy_prediction.ui.pages.overview import render_overview
from bankruptcy_prediction.ui.session_state import get_score_session

PAGE_NAMES = (
    "總覽與模型資訊",
    "批次評分",
    "個案解釋",
    "模型效能",
)


def render_app(client: ApiClient | None = None, *, streamlit_module: Any | None = None) -> None:
    """Render the navigation shell; page actions are implemented by later UI tasks."""
    if streamlit_module is None:
        import streamlit as streamlit_module
    st = streamlit_module
    st.set_page_config(page_title="企業破產預測系統", layout="wide")
    st.title("企業破產預測系統")
    st.caption("模型信心分數僅供分析優先順序使用。")
    api = client or ApiClient()
    try:
        health = api.health()
    except ApiClientError as exc:
        st.error(str(exc))
        st.info("請先啟動本機 FastAPI（127.0.0.1:8000），再重新整理頁面。")
        health = {"active_model_loaded": False}
    if not health.get("active_model_loaded", False):
        st.warning("目前沒有可用的 active model；批次評分控制已停用。")
    selected = st.radio("頁面", PAGE_NAMES, horizontal=True)
    st.session_state.setdefault("selected_page", selected)
    st.session_state["selected_page"] = selected
    if selected == PAGE_NAMES[0]:
        try:
            render_overview(api.model(), st)
        except ApiClientError as exc:
            st.error(str(exc))
    elif selected == PAGE_NAMES[1]:
        payload = render_batch_scoring(
            st,
            api,
            model_loaded=bool(health.get("active_model_loaded", False)),
        )
        if payload is not None:
            session = get_score_session(st.session_state)
            render_scoring_results(
                st,
                api,
                payload,
                csv_bytes=session[1] if session else None,
                filename=session[2] if session else "input.csv",
            )
    elif selected == PAGE_NAMES[2]:
        session = get_score_session(st.session_state)
        payload = session[0] if session else None
        render_case_explanation(
            st,
            payload or {"rows": []},
            session[1] if session else None,
        )
    else:
        try:
            render_model_performance(api.model(), st)
        except ApiClientError as exc:
            st.error(str(exc))


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
