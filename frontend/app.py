"""
frontend/app.py
----------------
Streamlit UI for DocuMind. Talks to the FastAPI backend over HTTP so the
two are fully decoupled -- the backend can be deployed separately (e.g.
behind a client's auth layer) and reused by other clients.

Run (after starting the backend, see README):
    streamlit run frontend/app.py
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("DOCUMIND_BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuMind — Document QA", page_icon="📄", layout="wide")

st.title("📄 DocuMind")
st.caption("Ask questions about your own documents. Every answer is grounded in — "
           "and cited back to — the exact source excerpt it came from.")

# ---------------------------------------------------------------------
# Backend connectivity check
# ---------------------------------------------------------------------
def backend_healthy() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.ok
    except requests.RequestException:
        return False


if not backend_healthy():
    st.error(
        f"⚠️ Can't reach the DocuMind backend at `{BACKEND_URL}`. "
        "Start it with `uvicorn main:app --reload --port 8000` from the `backend/` folder."
    )
    st.stop()

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
            files_payload = [
                ("files", (f.name, f.getvalue(), "application/octet-stream"))
                for f in uploaded_files
            ]
            resp = requests.post(f"{BACKEND_URL}/documents/upload", files=files_payload)
        if resp.ok:
            for item in resp.json():
                st.success(f"✅ {item['filename']}: {item['chunks_added']} chunks indexed")
        else:
            st.error(f"Upload failed: {resp.text}")

    st.divider()

    docs_resp = requests.get(f"{BACKEND_URL}/documents")
    if docs_resp.ok:
        docs_data = docs_resp.json()
        st.metric("Chunks indexed", docs_data["total_chunks"])
        if docs_data["sources"]:
            st.write("**Indexed documents:**")
            for src in docs_data["sources"]:
                st.write(f"- {src}")
        else:
            st.info("No documents indexed yet.")

    if st.button("🗑️ Clear all documents"):
        requests.delete(f"{BACKEND_URL}/documents")
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
    st.session_state.history.append({"question": question, "answer": "", "citations": []})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            resp = requests.post(f"{BACKEND_URL}/ask", json={"question": question, "top_k": top_k})
        if resp.ok:
            data = resp.json()
            st.write(data["answer"])
            if data["citations"]:
                with st.expander(f"📎 {len(data['citations'])} source excerpt(s)"):
                    for c in data["citations"]:
                        page_info = f", page {c['page']}" if c.get("page") not in (None, -1) else ""
                        st.markdown(f"**{c['source']}{page_info}**")
                        st.caption(c["excerpt"])
                        st.divider()
            st.session_state.history[-1] = {
                "question": question, "answer": data["answer"], "citations": data["citations"],
            }
        else:
            error_msg = resp.json().get("detail", resp.text)
            st.error(error_msg)
            st.session_state.history[-1]["answer"] = f"Error: {error_msg}"
