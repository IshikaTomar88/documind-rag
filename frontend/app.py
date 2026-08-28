"""
app.py
"""

from __future__ import annotations

import os

import streamlit as st

from ingest import ingest_any
from rag import answer_question
from vectorstore import VectorStore

st.set_page_config(page_title="DocuMind — Document QA", page_icon="📄", layout="wide")

st.title("📄 DocuMind")
st.caption(
    "Ask questions about your own documents. Every answer is grounded in — "
    "and cited back to — the exact source excerpt it came from."
)


# ---------------------------------------------------------------------
# Vector store: created once per (embedding backend) and reused across
# reruns/questions instead of being rebuilt on every interaction. This
# matters a lot for the "local" embedding backend (loads a multi-hundred-
# MB transformer model) and still helps "tfidf"/"openai".
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Setting up the document index...")
def get_store(embedding_backend: str) -> VectorStore:
    return VectorStore(backend=embedding_backend)


embedding_backend = os.environ.get("EMBEDDING_BACKEND", "tfidf").lower()
store = get_store(embedding_backend)

available_llm_backends = ["extractive"]
if os.environ.get("ANTHROPIC_API_KEY"):
    available_llm_backends.append("anthropic")
if os.environ.get("OPENAI_API_KEY"):
    available_llm_backends.append("openai")

default_llm_backend = os.environ.get("LLM_BACKEND", "extractive").lower()
if default_llm_backend not in available_llm_backends:
    default_llm_backend = "extractive"

# ---------------------------------------------------------------------
# Sidebar: document management
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Document library")
    st.caption(f"Embedding backend: `{embedding_backend}`")

    uploaded_files = st.file_uploader(
        "Upload PDFs or text files", type=["pdf", "txt", "md"], accept_multiple_files=True,
    )

    if uploaded_files and st.button("⬆️ Ingest documents", type="primary"):
        with st.spinner("Reading, chunking, and embedding your documents..."):
            for f in uploaded_files:
                try:
                    chunks = ingest_any(f.getvalue(), f.name)
                    n = store.add_chunks(chunks)
                    st.success(f"✅ {f.name}: {n} chunks indexed")
                except ValueError as e:
                    st.error(f"❌ {f.name}: {e}")
                except Exception as e:  # noqa: BLE001 - surface any parse/index failure to the user
                    st.error(f"❌ {f.name}: unexpected error — {e}")

    st.divider()

    st.metric("Chunks indexed", store.stats()["total_chunks"])
    sources = store.list_sources()
    if sources:
        st.write("**Indexed documents:**")
        for src in sources:
            st.write(f"- {src}")
    else:
        st.info("No documents indexed yet.")

    if st.button("🗑️ Clear all documents"):
        store.reset()
        st.rerun()

    st.divider()
    top_k = st.slider("Excerpts to retrieve per question", 1, 10, 4)

    if len(available_llm_backends) > 1:
        llm_backend = st.selectbox(
            "Answer generation",
            available_llm_backends,
            index=available_llm_backends.index(default_llm_backend),
            help=(
                "'extractive' needs no API key and just returns the most "
                "relevant sentences verbatim. 'anthropic'/'openai' send the "
                "retrieved excerpts to that provider's chat model."
            ),
        )
    else:
        llm_backend = "extractive"
        st.caption(
            "Answer generation: `extractive` (no ANTHROPIC_API_KEY / "
            "OPENAI_API_KEY set — set one as an environment variable to "
            "enable LLM-written answers)."
        )

# ---------------------------------------------------------------------
# Main: chat-style Q&A
# ---------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


def _render_citations(citations: list[dict]) -> None:
    with st.expander(f"📎 {len(citations)} source excerpt(s)"):
        for c in citations:
            page_info = f", page {c['page']}" if c.get("page") not in (None, -1) else ""
            st.markdown(f"**{c['source']}{page_info}**")
            st.caption(c["excerpt"])
            st.divider()


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["citations"]:
            _render_citations(turn["citations"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.history.append({"question": question, "answer": "", "citations": []})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if store.stats()["total_chunks"] == 0:
            error_msg = "No documents have been uploaded yet. Add some in the sidebar first."
            st.warning(error_msg)
            st.session_state.history[-1]["answer"] = error_msg
        else:
            with st.spinner("Searching documents and generating an answer..."):
                try:
                    result = answer_question(store, question, top_k=top_k, llm_backend=llm_backend)
                except Exception as e:  # noqa: BLE001 - never let a retrieval/generation error crash the UI
                    result = {"answer": f"Something went wrong answering that: {e}", "citations": []}

            st.write(result["answer"])
            if result["citations"]:
                _render_citations(result["citations"])
            st.session_state.history[-1] = {
                "question": question, "answer": result["answer"], "citations": result["citations"],
            }
