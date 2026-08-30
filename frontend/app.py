"""
DocuMind Enterprise — Document Intelligence Engine
---------------------------------------------------
Real RAG pipeline: extract -> chunk (per page) -> embed -> cache -> retrieve top-k -> cite.

Requirements:
    pip install streamlit pypdf google-genai numpy

Run:
    streamlit run documind_app.py

Set your key either in .streamlit/secrets.toml as GEMINI_API_KEY, or paste it
into the sidebar at runtime.
"""

import hashlib
import io
import re
import time
from collections import defaultdict

import numpy as np
import streamlit as st
from pypdf import PdfReader
from google import genai

# --------------------------------------------------------------------------
# Page config & styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuMind Enterprise - Document Intelligence Engine",
    page_icon="💼",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin-bottom: 0px; }
    .sub-title { color: #475569; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .src-chip { display:inline-block; background:#EEF2FF; color:#3730A3; border-radius:6px;
                padding:2px 8px; font-size:0.78rem; margin:2px 4px 2px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">💼 DocuMind Enterprise AI Workspace</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Upload client documents and query them with grounded, cited answers.</p>',
    unsafe_allow_html=True,
)

EMBED_MODEL = "gemini-embedding-001"
GEN_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.5-flash", "gemini-3.7-flash"]

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_store" not in st.session_state:
    # hash -> {"name": str, "chunks": [dict], "vectors": np.ndarray}
    st.session_state.index_store = {}


# --------------------------------------------------------------------------
# Core RAG helpers
# --------------------------------------------------------------------------
def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    text = clean_text(text)
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_pdf_pages(data: bytes):
    pages = []
    reader = PdfReader(io.BytesIO(data))
    for i, page in enumerate(reader.pages):
        try:
            pages.append((i + 1, page.extract_text() or ""))
        except Exception:
            pages.append((i + 1, ""))
    return pages


def build_chunks_for_file(name: str, data: bytes, chunk_size: int, overlap: int):
    chunks = []
    if name.lower().endswith(".pdf"):
        for page_num, text in extract_pdf_pages(data):
            for piece in chunk_text(text, chunk_size, overlap):
                chunks.append({"file": name, "page": page_num, "text": piece})
    else:
        text = data.decode("utf-8", errors="ignore")
        for piece in chunk_text(text, chunk_size, overlap):
            chunks.append({"file": name, "page": None, "text": piece})
    return chunks


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return genai.Client(api_key=api_key)


def embed_texts(client, texts, show_progress=True):
    """Returns (matrix, kept_indices). Failed items are skipped, not zero-filled."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32), []

    vectors, kept, failed = [], [], 0
    bar = st.progress(0.0) if show_progress else None
    for i, t in enumerate(texts):
        vec = None
        for attempt in range(3):
            try:
                resp = client.models.embed_content(model=EMBED_MODEL, contents=t[:8000])
                vec = np.array(resp.embeddings[0].values, dtype=np.float32)
                break
            except Exception:
                time.sleep(1.2 * (attempt + 1))
        if vec is not None:
            vectors.append(vec)
            kept.append(i)
        else:
            failed += 1
        if bar:
            bar.progress((i + 1) / len(texts))
    if bar:
        bar.empty()
    if failed:
        st.warning(f"{failed} chunk(s) could not be embedded after retries and were skipped.")

    matrix = np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)
    return matrix, kept


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, chunks: list, k: int):
    if matrix.shape[0] == 0:
        return []
    denom = (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)) + 1e-8
    sims = (matrix @ query_vec) / denom
    top_idx = np.argsort(-sims)[:k]
    return [(chunks[i], float(sims[i])) for i in top_idx]


def representative_sample(chunks: list, per_file: int = 4, max_total: int = 40):
    """Broad, stratified sample across files/pages — used for summary-style
    quick actions where narrow semantic similarity to the query is the wrong tool."""
    by_file = defaultdict(list)
    for c in chunks:
        by_file[c["file"]].append(c)
    sample = []
    for _, cs in by_file.items():
        step = max(1, len(cs) // per_file)
        sample.extend(cs[::step][:per_file])
    return sample[:max_total]


def format_source(c: dict) -> str:
    return f"{c['file']}" + (f", p.{c['page']}" if c.get("page") else "")


def build_context(pairs_or_chunks, with_scores=False):
    blocks = []
    for item in pairs_or_chunks:
        c, score = item if with_scores else (item, None)
        blocks.append(f"[Source: {format_source(c)}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def active_corpus(active_hashes):
    """Concatenate cached chunks/vectors for the files currently uploaded."""
    all_chunks, mats = [], []
    for h in active_hashes:
        entry = st.session_state.index_store.get(h)
        if not entry:
            continue
        all_chunks.extend(entry["chunks"])
        if entry["vectors"].shape[0]:
            mats.append(entry["vectors"])
    matrix = np.vstack(mats) if mats else np.zeros((0, 1), dtype=np.float32)
    return all_chunks, matrix


SYSTEM_INSTRUCTION = (
    "You are DocuMind Enterprise, a precise document-reasoning assistant for corporate "
    "clients. You will be given CONTEXT CHUNKS pulled from the client's uploaded documents, "
    "each labeled with its source file (and page number, if it's a PDF). Rules:\n"
    "1. Answer using ONLY the information in the context chunks below — do not use outside "
    "knowledge.\n"
    "2. After every factual claim, cite the source inline like (filename, p.X).\n"
    "3. If the answer is not present in the context, say clearly that it was not found in "
    "the indexed documents — do not guess.\n"
    "4. Be concise and professional."
)


def answer_freeform(client, gen_model, query, history_text, chunks, matrix, k):
    q_matrix, kept = embed_texts(client, [query], show_progress=False)
    if not kept:
        return "Could not embed the question — please retry.", []
    hits = cosine_topk(q_matrix[0], matrix, chunks, k=k)
    context = build_context(hits, with_scores=True)
    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\nCONTEXT CHUNKS:\n{context}\n\n"
        f"CONVERSATION HISTORY:\n{history_text}\n\nQUESTION: {query}"
    )
    resp = client.models.generate_content(model=gen_model, contents=prompt)
    return resp.text, hits


def answer_broad(client, gen_model, instruction, chunks):
    sample = representative_sample(chunks)
    context = build_context(sample, with_scores=False)
    prompt = f"{SYSTEM_INSTRUCTION}\n\nCONTEXT CHUNKS (broad sample across all documents):\n{context}\n\nTASK: {instruction}"
    resp = client.models.generate_content(model=gen_model, contents=prompt)
    pairs = [(c, None) for c in sample]
    return resp.text, pairs


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    if st.button("🆕 Start New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(
        "Wipes chat memory completely — nothing from the previous conversation is "
        "sent to the model again. Your indexed documents stay loaded (no re-uploading)."
    )

    st.markdown("---")
    st.header("🔑 Configuration")
    st.caption(
        "This app calls **Google's Gemini API**, not OpenAI/ChatGPT. Get a free key "
        "(starts with `AIza...`) at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)."
    )
    default_key = st.secrets.get("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Google Gemini API Key", value=default_key, type="password"
    ).strip()

    if api_key and not api_key.startswith("AIza"):
        st.error(
            "That doesn't look like a Gemini key. OpenAI/ChatGPT keys (`sk-...`) and "
            "other providers' keys won't work here — this app only calls the Gemini API."
        )
        st.stop()

    gen_model = st.selectbox("Generation model", GEN_MODELS, index=0)

    with st.expander("⚙️ Advanced settings"):
        chunk_size = st.slider("Chunk size (characters)", 400, 2000, 900, 100)
        overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, 25)
        top_k = st.slider("Chunks retrieved per question", 2, 15, 6, 1)

    st.divider()
    st.header("📁 Client Document Vault")
    uploaded_files = st.file_uploader(
        "Upload multi-page PDFs or text files",
        type=["pdf", "txt", "csv", "md"],
        accept_multiple_files=True,
    )

    active_hashes = []
    if uploaded_files:
        active_hashes = [file_hash(f.getvalue()) for f in uploaded_files]
        missing = [
            (f, h) for f, h in zip(uploaded_files, active_hashes)
            if h not in st.session_state.index_store
        ]

        if missing and not api_key:
            st.warning(f"⚠️ {len(missing)} new file(s) need indexing — add your API key to build the index.")
        elif missing and api_key:
            client = get_client(api_key)
            for f, h in missing:
                with st.spinner(f"Indexing {f.name}..."):
                    chunks = build_chunks_for_file(f.name, f.getvalue(), chunk_size, overlap)
                    if not chunks:
                        st.error(f"No extractable text found in {f.name}.")
                        continue
                    matrix, kept = embed_texts(client, [c["text"] for c in chunks])
                    chunks = [chunks[i] for i in kept]
                    st.session_state.index_store[h] = {
                        "name": f.name, "chunks": chunks, "vectors": matrix,
                    }
                st.success(f"Indexed {f.name} — {len(chunks)} chunk(s)")

        st.caption("Currently indexed in this session:")
        for h in active_hashes:
            entry = st.session_state.index_store.get(h)
            if entry:
                st.markdown(f"- **{entry['name']}** — {len(entry['chunks'])} chunks")

        if st.button("🔁 Rebuild index with current settings", use_container_width=True):
            for h in list(active_hashes):
                st.session_state.index_store.pop(h, None)
            st.rerun()

    st.divider()
    if st.session_state.get("confirm_clear_vault"):
        st.warning("Remove all indexed documents and chat history? This can't be undone.")
        cc1, cc2 = st.columns(2)
        if cc1.button("Yes, clear everything", use_container_width=True):
            st.session_state.index_store = {}
            st.session_state.messages = []
            st.session_state.confirm_clear_vault = False
            st.rerun()
        if cc2.button("Cancel", use_container_width=True):
            st.session_state.confirm_clear_vault = False
            st.rerun()
    else:
        if st.button("🗑️ Remove All Documents & Reset", use_container_width=True):
            st.session_state.confirm_clear_vault = True
            st.rerun()


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
if not uploaded_files:
    st.info("👈 Upload client documents in the sidebar to activate the DocuMind intelligence engine.")
    st.stop()

if not api_key:
    st.warning("⚠️ Please provide your Google Gemini API key in the sidebar to proceed.")
    st.stop()

client = get_client(api_key)
chunks, matrix = active_corpus(active_hashes)

if not chunks:
    st.info("Indexing in progress or no text could be extracted yet — check the sidebar.")
    st.stop()

st.caption(f"📚 {len(set(c['file'] for c in chunks))} document(s) · {len(chunks)} indexed chunks")
st.caption(
    "🔒 This conversation only ever sees messages from *this* session — no memory "
    "persists across browser sessions or leaks in from other users/clients."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📎 {len(message['sources'])} source(s) used"):
                for c, score in message["sources"]:
                    label = format_source(c)
                    score_txt = f" · similarity {score:.2f}" if score is not None else ""
                    st.markdown(f"<span class='src-chip'>{label}{score_txt}</span>", unsafe_allow_html=True)
                    st.caption(c["text"][:300] + ("…" if len(c["text"]) > 300 else ""))

st.markdown("##### ⚡ Executive Quick Actions")
col1, col2, col3 = st.columns(3)
quick_prompt, quick_mode = None, None
with col1:
    if st.button("📝 Executive Summary", use_container_width=True):
        quick_prompt = "Provide a comprehensive executive summary of the uploaded documents, highlighting core objectives and key takeaways."
        quick_mode = "broad"
with col2:
    if st.button("⚠️ Risk & Liability Analysis", use_container_width=True):
        quick_prompt = "Analyze the documents for potential risks, liabilities, or critical clauses. List them clearly with recommendations."
        quick_mode = "broad"
with col3:
    if st.button("📅 Dates & Action Items", use_container_width=True):
        quick_prompt = "Extract all critical deadlines, dates, financial figures, and action items mentioned in the documents."
        quick_mode = "broad"

user_query = st.chat_input("Ask anything about your documents...")
active_prompt = user_query if user_query else quick_prompt
mode = "freeform" if user_query else quick_mode

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Analyzing document corpus with {gen_model}..."):
            try:
                if mode == "broad":
                    output_text, sources = answer_broad(client, gen_model, active_prompt, chunks)
                else:
                    history_text = "\n".join(
                        f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages[-6:-1]
                    )
                    output_text, sources = answer_freeform(
                        client, gen_model, active_prompt, history_text, chunks, matrix, top_k
                    )

                st.markdown(output_text)
                if sources:
                    with st.expander(f"📎 {len(sources)} source(s) used"):
                        for c, score in sources:
                            label = format_source(c)
                            score_txt = f" · similarity {score:.2f}" if score is not None else ""
                            st.markdown(f"<span class='src-chip'>{label}{score_txt}</span>", unsafe_allow_html=True)
                            st.caption(c["text"][:300] + ("…" if len(c["text"]) > 300 else ""))

                st.session_state.messages.append(
                    {"role": "assistant", "content": output_text, "sources": sources}
                )
            except Exception as e:
                st.error(f"Execution failed: {e}")
