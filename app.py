"""Spain Paperwork Assistant: ask about NIE, empadronamiento, health card and degree recognition."""
from __future__ import annotations

import os
import time

import streamlit as st
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))
for key in ("GOOGLE_API_KEY", "GROQ_API_KEY", "MAX_QUESTIONS", "LLM_MODEL"):  # Streamlit Cloud keeps secrets outside the environment
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

from rag.answer import answer, cited  # noqa: E402  (after the environment is ready)
from rag.embeddings import embed  # noqa: E402

MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "10"))

EXAMPLES = [
    "¿Qué documentos necesito para empadronarme en Elche?",
    "How do I get the health card (SIP) in the Comunitat Valenciana?",
    "Como homologar meu diploma universitário estrangeiro na Espanha?",
    "What forms do I need to apply for the NIE from abroad?",
]

st.set_page_config(page_title="Spain Paperwork Assistant", page_icon="📄", layout="centered")
st.markdown(
    """<style>
    .block-container {padding-top: 2.2rem; max-width: 820px;}
    h1 {font-weight: 700;}
    .src {border: 1px solid #26262a; border-radius: 8px; padding: .6rem .9rem; margin: .4rem 0; background: #151517;}
    .src a {color: #ff5a5f; text-decoration: none;}
    .src small {color: #a2a2aa;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def _answer_cache() -> dict:
    return {}


def cached_answer(question: str):
    """Repeated questions cost no model quota. A failed (busy) answer is never cached."""
    store = _answer_cache()
    hit = store.get(question)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    result = answer(question, embed)
    if not result.busy:
        store[question] = (time.time(), result)
    return result


def render_sources(sources):
    for i, s in sources:
        st.markdown(
            f'<div class="src"><b>[{i}]</b> <a href="{s.url}" target="_blank">{s.title}</a><br>'
            f"<small>{s.topic} · page read on {s.fetched_at}</small></div>",
            unsafe_allow_html=True,
        )


if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.asked = 0

st.title("📄 Spain Paperwork Assistant")
st.caption("Ask about **NIE, empadronamiento, the health card (SIP) and degree recognition**, in Spanish, English or Portuguese. Every answer cites the official page it comes from.")

with st.sidebar:
    st.subheader("Try asking")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True, disabled=st.session_state.asked >= MAX_QUESTIONS):
            st.session_state.pending = ex
    st.divider()
    st.caption(
        "⚠️ General information from official pages, **not legal advice**. Rules change: always confirm on the "
        "official site. Built on free services, so each session is limited to "
        f"{MAX_QUESTIONS} questions."
    )
    st.caption(f"Questions used: {st.session_state.asked}/{MAX_QUESTIONS}")

for i, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})", expanded=i == len(st.session_state.history) - 1):
                render_sources(msg["sources"])

question = st.chat_input("Ask your question…", disabled=st.session_state.asked >= MAX_QUESTIONS)
question = question or st.session_state.pop("pending", None)

if st.session_state.asked >= MAX_QUESTIONS:
    st.info("You reached the question limit for this session (it protects the free AI quota). Reload the page to start again.")

if question and st.session_state.asked < MAX_QUESTIONS:
    st.session_state.asked += 1
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the official pages…"):
            result = cached_answer(question)
    shown = cited(result.text, result.sources) if result.sources else []
    st.session_state.history.append({"role": "assistant", "content": result.text, "sources": shown})
    st.rerun()  # redraw so the counter and the disabled input reflect this question
