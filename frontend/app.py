"""
app.py
"""

from __future__ import annotations

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


@st.cache_resource
def get_store() -> VectorStore:
    # st.cache_resource keeps one VectorStore alive for the life of the
    # app process, instead of re-opening the on-disk Chroma index (and
    # re-loading the TF-IDF vectorizer) on every single rerun.
    return VectorStore()


store = get_store()

# ---------------------------------------------------------------------
# Sidebar: document management
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Document library")

    uploaded_files = st.file_uploader(
        "Upload PDFs or text files", type=["pdf", "txt", "md"], accept_multiple_files=True,
    )

    if uploaded_files and st.button("⬆️ Ingest documents", type="primary"):
        with st.spinner("Reading, chunking, and embedding your documents..."):
            for f in uploaded_files:
                try:
                    chunks = ingest_any(f.getvalue(), f.name)
                except ValueError as e:
                    st.error(f"❌ {f.name}: {e}")
                    continue
                n = store.add_chunks(chunks)
                st.success(f"✅ {f.name}: {n} chunks indexed")

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

# ---------------------------------------------------------------------
# Main: chat-style Q&A
# ---------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["citations"]:
            with st.expander(f"📎 {len(turn['citations'])} source excerpt(s)"):
                for c in turn["citations"]:
                    page_info = f", page {c['page']}" if c.get("page") not in (None, -1) else ""
                    st.markdown(f"**{c['source']}{page_info}**")
                    st.caption(c["excerpt"])
                    st.divider()

question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if store.stats()["total_chunks"] == 0:
            error_msg = "No documents have been uploaded yet. Add a file in the sidebar first."
            st.error(error_msg)
            st.session_state.history.append({"question": question, "answer": error_msg, "citations": []})
        else:
            with st.spinner("Searching documents and generating an answer..."):
                result = answer_question(store, question, top_k=top_k)
            st.write(result["answer"])
            if result["citations"]:
                with st.expander(f"📎 {len(result['citations'])} source excerpt(s)"):
                    for c in result["citations"]:
                        page_info = f", page {c['page']}" if c.get("page") not in (None, -1) else ""
                        st.markdown(f"**{c['source']}{page_info}**")
                        st.caption(c["excerpt"])
                        st.divider()
            st.session_state.history.append({
                "question": question,
                "answer": result["answer"],
                "citations": result["citations"],
            })
