"""LexiMind AI — Streamlit application entry point."""

from pathlib import Path

import streamlit as st

from ui.pages import PAGE_BY_LABEL, PAGES, render_page
from utils.file_handler import extract_document
from utils.helpers import initialize_session_state

st.set_page_config(
    page_title="LexiMind AI | NLP Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_file = Path(__file__).parent / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

initialize_session_state(st.session_state)
st.session_state.setdefault("page", PAGES[0][0])
st.session_state.setdefault("source_text", "")
st.session_state.setdefault("module_results", {})


def _navigate_to(label: str) -> None:
    st.session_state["page"] = label


with st.sidebar:
    st.markdown("# 🧠 LexiMind AI")
    st.caption("Understand Language. Unlock Intelligence.")
    st.divider()
    selected_label = st.radio("WORKSPACE", [item[0] for item in PAGES], key="page", label_visibility="collapsed")
    st.divider()
    st.markdown("#### Document input")
    uploaded_file = st.file_uploader("Load TXT or PDF", type=["txt", "pdf"], key="document_file")
    if st.button("Load document into workspace", key="load_document", width="stretch"):
        if uploaded_file is None:
            st.warning("Choose a TXT or PDF file first.")
        else:
            try:
                st.session_state["source_text"] = extract_document(uploaded_file.name, uploaded_file.getvalue())
                st.session_state["loaded_filename"] = uploaded_file.name
                st.success("Document loaded. Open a module to analyze it.")
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
    st.divider()
    st.markdown("#### Academic project")
    st.caption("B.Tech AIML · 5th Semester")
    st.caption("RGPV University")
    st.caption("Runs locally · No API keys")

render_page(PAGE_BY_LABEL[selected_label], _navigate_to)

st.divider()
st.caption("LexiMind AI · All-in-One Natural Language Processing Platform · B.Tech AIML | RGPV")
