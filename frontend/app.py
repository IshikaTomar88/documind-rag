"""
DocuMind Enterprise — Document Intelligence Engine
---------------------------------------------------
Hybrid-privacy RAG pipeline:
  extract -> chunk (per page) -> embed LOCALLY (fastembed / ONNX runtime, no
  key, no PyTorch, no network call after first model download — documents
  never leave this machine for indexing) -> cache -> retrieve top-k by cosine
  similarity -> generate the cited answer via a cloud LLM (Gemini). Only the
  final retrieved snippets + your question ever reach the cloud.

Requirements:
    pip install streamlit pypdf google-genai numpy fastembed

Run:
    streamlit run documind_app.py
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
from fastembed import TextEmbedding

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

LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GEN_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_store" not in st.session_state:
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


@st.cache_resource(show_spinner="Loading local embedding model (first run only)...")
def get_embedder():
    return TextEmbedding(model_name=LOCAL_EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return genai.Client(api_key=api_key)


def embed_texts(texts, show_progress=True):
    if not texts:
        return np.zeros((0, 1), dtype=np.float32), [], None
    try:
        embedder = get_embedder()
        bar = st.progress(0.0, text="Embedding locally...") if show_progress else None
        vectors = np.array(list(embedder.embed(texts)), dtype=np.float32)
        if bar:
            bar.progress(1.0)
            bar.empty()
        return vectors, list(range(len(texts))), None
    except Exception as e:
        return np.zeros((0, 1), dtype=np.float32), [], str(e)


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, chunks: list, k: int):
    if matrix.shape[0] == 0:
        return []
    denom = (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)) + 1e-8
    sims = (matrix @ query_vec) / denom
    top_idx = np.argsort(-sims)[:k]
    return [(chunks[i], float(sims[i])) for i in top_idx]


def representative_sample(chunks: list, per_file: int = 4, max_total: int = 40):
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
    "1. Answer using ONLY the information in the context chunks below — do not use outside knowledge.\n"
    "2. After every factual claim, cite the source inline like (filename, p.X).\n"
    "3. If the answer is not present in the context, say clearly that it was not found in the indexed documents.\n"
    "4. Be concise and professional."
)


def generate_with_retry(client, gen_model, prompt, max_retries=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=gen_model, contents=prompt)
            return resp.text, None
        except Exception as e:
            msg = str(e)
            last_error = msg
            transient = any(code in msg for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
            if not transient or attempt == max_retries - 1:
                break
            time.sleep(2 * (attempt + 1))
    return None, last_error


def answer_freeform(client, gen_model, query, history_text, chunks, matrix, k):
    q_matrix, kept, err = embed_texts([query], show_progress=False)
    if not kept:
        detail = f" Error: {err}" if err else ""
        return f"Could not process the question.{detail}", []
    hits = cosine_topk(q_matrix[0], matrix, chunks, k=k)
    context = build_context(hits, with_scores=True)
    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\nCONTEXT CHUNKS:\n{context}\n\n"
        f"CONVERSATION HISTORY:\n{history_text}\n\nQUESTION: {query}"
    )
    text, error = generate_with_retry(client, gen_model, prompt)
    if error:
        raise RuntimeError(error)
    return text, hits


def answer_broad(client, gen_model, instruction, chunks):
    sample = representative_sample(chunks)
    context = build_context(sample, with_scores=False)
    prompt = f"{SYSTEM_INSTRUCTION}\n\nCONTEXT CHUNKS (broad sample across all documents):\n{context}\n\nTASK: {instruction}"
    text, error = generate_with_retry(client, gen_model, prompt)
    if error:
        raise RuntimeError(error)
    pairs = [(c, None) for c in sample]
    return text, pairs


# --------------------------------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------------------------------
with st.sidebar:
    if st.button("🆕 Start New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.header("🔑 Configuration")
    
    # Safely load key from Streamlit secrets if configured, otherwise provide input box
    default_key = ""
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    api_key = st.text_input(
        "Google Gemini API Key", value=default_key, type="password",
        help="Get a free key at aistudio.google.com/apikey",
    ).strip()

    if api_key.startswith("sk-"):
        st.error(
            "That looks like an OpenAI key (`sk-...`), not a Gemini key. "
            "Get a Gemini key at aistudio.google.com/apikey."
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

        if missing:
            for f, h in missing:
                with st.spinner(f"Indexing {f.name} locally..."):
                    chunks = build_chunks_for_file(f.name, f.getvalue(), chunk_size, overlap)
                    if not chunks:
                        st.error(f"No extractable text found in {f.name}.")
                        continue
                    matrix, kept, err = embed_texts([c["text"] for c in chunks])
                    chunks = [chunks[i] for i in kept]

                if not chunks:
                    st.error(f"❌ Local indexing failed for {f.name}.")
                    if err:
                        st.code(err, language=None)
                else:
                    st.session_state.index_store[h] = {
                        "name": f.name, "chunks": chunks, "vectors": matrix,
                    }
                    st.success(f"Indexed {f.name} — {len(chunks)} chunk(s)")

        st.caption("Currently indexed in this session:")
        for h in active_hashes:
            entry = st.session_state.index_store.get(h)
            if entry:
                st.markdown(f"- **{entry['name']}** — {len(entry['chunks'])} chunks")

    st.divider()
    if st.button("🗑️ Remove All Documents & Reset", use_container_width=True):
        st.session_state.index_store = {}
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------
# Main area execution
# --------------------------------------------------------------------------
if not uploaded_files:
    st.info("👈 Upload client documents in the sidebar to activate the DocuMind engine.")
    st.stop()

chunks, matrix = active_corpus(active_hashes)

if not chunks:
    st.info("Indexing in progress or no text could be extracted yet.")
    st.stop()

if not api_key:
    st.warning("⚠️ Documents are indexed locally. Please enter your Gemini API key in the sidebar to ask questions.")
    st.stop()

client = get_client(api_key)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📎 {len(message['sources'])} source(s) used"):
                for c, score in message["sources"]:
                    label = format_source(c)
                    score_txt = f" · similarity {score:.2f}" if score is not None else ""
                    st.markdown(f"<span class='src-chip'>{label}{score_txt}</span>", unsafe_allow_html=True)

st.markdown("##### ⚡ Executive Quick Actions")
col1, col2, col3 = st.columns(3)
quick_prompt, quick_mode = None, None
with col1:
    if st.button("📝 Executive Summary", use_container_width=True):
        quick_prompt = "Provide a comprehensive executive summary of the uploaded documents."
        quick_mode = "broad"
with col2:
    if st.button("⚠️ Risk Analysis", use_container_width=True):
        quick_prompt = "Analyze the documents for potential risks, liabilities, or critical clauses."
        quick_mode = "broad"
with col3:
    if st.button("📅 Dates & Action Items", use_container_width=True):
        quick_prompt = "Extract all critical deadlines, dates, and action items mentioned."
        quick_mode = "broad"

user_query = st.chat_input("Ask anything about your documents...")
active_prompt = user_query if user_query else quick_prompt
mode = "freeform" if user_query else quick_mode

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Analyzing with {gen_model}..."):
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

                st.session_state.messages.append(
                    {"role": "assistant", "content": output_text, "sources": sources}
                )
            except Exception as e:
                st.error(f"Execution failed: {str(e)}")
