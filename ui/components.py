"""Reusable Streamlit components shared across module pages."""

import streamlit as st

from utils.helpers import history_csv, history_report, record_analysis


def set_source_text(text: str) -> None:
    """Set workspace text from button callbacks before widgets are recreated."""
    st.session_state["source_text"] = text


def save_analysis(page: str, text: str, result: dict, summary: str) -> None:
    st.session_state["module_results"][page] = {"input": text, "result": result}
    record_analysis(st.session_state, page, text, summary)


def show_saved_result(page: str) -> dict | None:
    saved = st.session_state["module_results"].get(page)
    if saved:
        st.caption("Most recent result — run the analysis again to update it.")
        return saved["result"]
    return None


def render_text_workspace(sample_text: str) -> str:
    """Render the common editor and its real sample/clear actions."""
    st.markdown("#### Text workspace")
    if st.session_state.get("loaded_filename"):
        st.caption(f"Loaded document: {st.session_state['loaded_filename']}")
    sample_col, clear_col, _ = st.columns([1.2, 1, 5])
    if sample_col.button("Use example", key="use_example", width="stretch"):
        st.session_state["source_text"] = sample_text
    if clear_col.button("Clear text", key="clear_text", width="stretch"):
        st.session_state["source_text"] = ""
        st.session_state.pop("loaded_filename", None)
    return st.text_area(
        "Text to analyze",
        key="source_text",
        height=190,
        placeholder="Paste text here or load a TXT/PDF document from the sidebar...",
        label_visibility="collapsed",
    )


def render_downloads() -> None:
    history = st.session_state["analysis_history"]
    if not history:
        return
    first, second = st.columns(2)
    first.download_button(
        "Download result as TXT",
        data=history_report(history),
        file_name="leximind-analysis-report.txt",
        mime="text/plain",
        width="stretch",
    )
    second.download_button(
        "Download history as CSV",
        data=history_csv(history),
        file_name="leximind-analysis-history.csv",
        mime="text/csv",
        width="stretch",
    )


def render_module_heading(title: str, description: str) -> None:
    st.markdown(f"<div class='eyebrow'>Interactive module</div><h1>{title}</h1>", unsafe_allow_html=True)
    st.caption(description)
