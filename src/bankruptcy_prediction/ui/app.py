"""Streamlit application shell with four API-backed pages."""

from __future__ import annotations

from typing import Any

from bankruptcy_prediction.ui.api_client import ApiClient, ApiClientError
from bankruptcy_prediction.ui.components.scoring_results import render_scoring_results
from bankruptcy_prediction.ui.pages.batch_scoring import render_batch_scoring
from bankruptcy_prediction.ui.pages.case_explanation import render_case_explanation
from bankruptcy_prediction.ui.pages.model_performance import render_model_performance
from bankruptcy_prediction.ui.pages.overview import render_overview
from bankruptcy_prediction.ui.pages.research_dashboard import render_research_dashboard
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
    theme = st.selectbox("介面主題 / Theme", ("深色 Dark", "淺色 Light"), key="dashboard_theme")
    dark = theme != "淺色 Light"
    bg, fg, panel, border = (
        ("#0c1222", "#eff6ff", "#151f35", "#304260")
        if dark
        else ("#f6f9fe", "#17253c", "#ffffff", "#dce5f3")
    )
    st.markdown(
        f"""<style>
      .stApp {{ background: {bg}; color: {fg}; }}
      [data-testid='stMetric'] {{ background:{panel}; padding:1rem; border:1px solid {border}; border-radius:16px; }}
      [data-testid='stMetricLabel'] {{ color:{fg}; }}
      h1,h2,h3 {{ letter-spacing:-0.025em; }}
    </style>""",
        unsafe_allow_html=True,
    )
    st.title("企業破產預測系統")
    st.caption("模型信心分數僅供分析優先順序使用。")
    api = client or ApiClient()
    try:
        health = api.health()
    except ApiClientError as exc:
        st.error(str(exc))
        st.info("請先啟動本機 FastAPI（127.0.0.1:8000），再重新整理頁面。")
        health = {"active_model_loaded": False}
    model_summary: dict[str, Any] | None = None
    if health.get("active_model_loaded", False):
        try:
            model_summary = api.model()
        except ApiClientError as exc:
            st.error(str(exc))
            health = {"active_model_loaded": False}
    if not health.get("active_model_loaded", False):
        st.warning("目前沒有可用的 active model；批次評分控制已停用。")
    selected = st.radio("頁面", PAGE_NAMES, horizontal=True)
    st.session_state.setdefault("selected_page", selected)
    st.session_state["selected_page"] = selected
    if selected == PAGE_NAMES[0]:
        render_research_dashboard(st)
        if model_summary is not None:
            render_overview(model_summary, st)
    elif selected == PAGE_NAMES[1]:
        payload = render_batch_scoring(
            st,
            api,
            model_loaded=bool(health.get("active_model_loaded", False)),
            default_threshold=(
                float(model_summary["default_threshold"])
                if model_summary is not None
                else 0.5
            ),
        )
        session = get_score_session(st.session_state)
        if session is not None:
            render_scoring_results(
                st,
                api,
                session[0],
                csv_bytes=session[1],
                filename=session[2],
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
        render_research_dashboard(st)
        if model_summary is not None:
            render_model_performance(model_summary, st)


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
