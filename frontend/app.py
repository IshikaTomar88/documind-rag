"""
DocuMind Enterprise — Document Intelligence
--------------------------------------------
Business problem this solves: a client has a pile of contracts/manuals/
policies and needs answers fast instead of reading 300 pages manually.

Two tools, matched to two different kinds of question:

1. Quick Word/Phrase Search (sidebar) — for "is X mentioned, and where?"
   This is answered with plain Python string matching, NOT the LLM.
   It's instant, free, and can never hallucinate a match that isn't
   really there — which matters for exactly this kind of yes/no,
   contract-clause-checking question.

2. Chat (main panel) — for "summarize this", "explain the termination
   clause", "translate section 3 into Spanish", open-ended questions
   that genuinely need language understanding. This uses Gemini with
   real multi-turn memory (a primed chat session), not a fresh one-shot
   prompt per question, so follow-up questions actually work and the
   whole conversation stays visible.

Honest limitation (see README): this sends the full text of selected
documents as context on every question rather than doing chunked
retrieval (RAG). For a handful of documents up to a few hundred pages
this works well and keeps the code simple. For a large, ever-growing
document library, a real vector-search pipeline (chunk + embed +
retrieve) would be cheaper per question and scale further — that's a
different, larger build, not a drop-in tweak to this one.
"""

from __future__ import annotations

import hashlib
import io
import re

import streamlit as st
from pypdf import PdfReader

st.set_page_config(
    page_title="DocuMind Enterprise - Document Intelligence",
    page_icon="🏢",
    layout="wide",
)

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin-bottom: 0px; }
    .sub-title { color: #475569; font-size: 1.05rem; margin-bottom: 2rem; }
    .response-container { background-color: #F8FAFC; border: 1px solid #CBD5E1; padding: 24px; border-radius: 10px; margin-top: 15px; }
    .search-hit { background-color: #FEF9C3; border: 1px solid #FDE68A; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px; }
    .warn-box { background-color: #FEF2F2; border: 1px solid #FCA5A5; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏢 DocuMind Enterprise Document Chat</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Upload contracts, manuals, HR handbooks, or messy PDFs. '
    'Search for exact terms instantly, or chat for summaries, explanations, and translations.</p>',
    unsafe_allow_html=True,
)

MODEL_NAME = "gemini-2.5-flash"
# Rough guide only (not a hard token count): past this, a single question
# is sending a LOT of text to the model every turn. Shown to the user so
# cost/latency isn't a surprise, not enforced as a hard limit.
LARGE_CORPUS_CHAR_WARNING = 400_000


# ---------------------------------------------------------------------
# Parsing (cached — a document is only ever parsed once per file content,
# not on every question)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_document(file_bytes: bytes, filename: str) -> list[dict]:
    """Returns a list of {'page': int|None, 'text': str} — one entry per
    PDF page, or a single entry for plain text/CSV/markdown files."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": i, "text": text})
        return pages
    text = file_bytes.decode("utf-8", errors="ignore")
    return [{"page": None, "text": text}]


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:12]


# ---------------------------------------------------------------------
# Deterministic word/phrase search — no LLM involved, can't hallucinate
# ---------------------------------------------------------------------
def search_term(corpus: dict[str, list[dict]], term: str, whole_word: bool, context_chars: int = 90) -> list[dict]:
    if not term.strip():
        return []
    pattern = re.escape(term.strip())
    if whole_word:
        pattern = rf"\b{pattern}\b"
    regex = re.compile(pattern, re.IGNORECASE)

    hits = []
    for filename, pages in corpus.items():
        for entry in pages:
            text = entry["text"]
            for m in regex.finditer(text):
                start = max(0, m.start() - context_chars)
                end = min(len(text), m.end() + context_chars)
                snippet = text[start:end].replace("\n", " ").strip()
                hits.append({
                    "file": filename,
                    "page": entry["page"],
                    "snippet": snippet,
                })
    return hits


# ---------------------------------------------------------------------
# Sidebar: config + document vault
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Enterprise Settings")
    api_key = st.text_input(
        "Google Gemini API Key", type="password",
        help="Enter your Gemini API key to activate the assistant.",
    )

    st.divider()
    st.header("📁 Client Document Vault")

    uploaded_files = st.file_uploader(
        "Upload multi-page PDFs or text files",
        type=["pdf", "txt", "csv", "md"],
        accept_multiple_files=True,
    )

    document_corpus: dict[str, list[dict]] = {}
    empty_files = []

    if uploaded_files:
        for file in uploaded_files:
            try:
                pages = parse_document(file.getvalue(), file.name)
                total_chars = sum(len(p["text"]) for p in pages)
                if total_chars == 0:
                    empty_files.append(file.name)
                else:
                    document_corpus[file.name] = pages
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

        if document_corpus:
            st.success(f"Indexed {len(document_corpus)} document(s)")
        if empty_files:
            st.markdown(
                f'<div class="warn-box">⚠️ No extractable text found in: '
                f'{", ".join(empty_files)}. This usually means it\'s a '
                f'scanned/image PDF — it would need OCR before this tool '
                f'can read it.</div>',
                unsafe_allow_html=True,
            )

    if document_corpus:
        st.divider()
        st.subheader("🔍 Quick Word/Phrase Search")
        st.caption("Instant exact-match search — doesn't use the AI, can't be wrong about whether a term appears.")
        search_query = st.text_input("Find in documents", key="search_box")
        whole_word = st.checkbox("Whole word only", value=True)
        if search_query:
            hits = search_term(document_corpus, search_query, whole_word)
            if hits:
                st.caption(f"{len(hits)} match(es)")
                for h in hits[:25]:
                    page_info = f", page {h['page']}" if h["page"] else ""
                    st.markdown(
                        f'<div class="search-hit"><b>{h["file"]}{page_info}</b><br>'
                        f'…{h["snippet"]}…</div>',
                        unsafe_allow_html=True,
                    )
                if len(hits) > 25:
                    st.caption(f"...and {len(hits) - 25} more matches not shown.")
            else:
                st.info("No matches found.")

        st.divider()
        st.subheader("💬 Chat scope")
        selected_files = st.multiselect(
            "Documents to include in chat context",
            options=list(document_corpus.keys()),
            default=list(document_corpus.keys()),
            help="Fewer documents = faster, cheaper, more focused answers.",
        )
        total_chars = sum(
            len(p["text"]) for name in selected_files for p in document_corpus[name]
        )
        st.caption(f"~{total_chars:,} characters in context")
        if total_chars > LARGE_CORPUS_CHAR_WARNING:
            st.warning(
                "This is a lot of text to send on every question — expect "
                "slower, more expensive responses. Consider narrowing the "
                "selection above to the documents actually relevant to "
                "what you're asking."
            )
    else:
        selected_files = []

# ---------------------------------------------------------------------
# Main workspace
# ---------------------------------------------------------------------
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # [{"role": "user"/"assistant", "text": ...}]
if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = None
if "primed_corpus_key" not in st.session_state:
    st.session_state.primed_corpus_key = None

if not uploaded_files:
    st.info("👈 Enter your Gemini API key and upload your client's documents in the sidebar to activate the system.")
elif not document_corpus:
    st.warning("None of the uploaded files had readable text. See the warning in the sidebar.")
elif not api_key:
    st.warning("⚠️ Please provide your Gemini API key in the sidebar to power the analysis.")
else:
    if not selected_files:
        st.info("👈 Select at least one document in the sidebar's 'Chat scope' to start chatting.")
        st.stop()

    # A new chat session is primed only when the API key or the selected
    # document set actually changes -- not on every rerun, and not by
    # resending the corpus on every question (that was the original
    # code's biggest cost/latency problem).
    corpus_key = file_hash((api_key + "|" + ",".join(sorted(selected_files))).encode())

    if st.session_state.primed_corpus_key != corpus_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            compiled_corpus = "\n\n".join(
                f"=== FILE: {name} ===\n" + "\n".join(
                    f"[Page {p['page']}]\n{p['text']}" if p["page"] else p["text"]
                    for p in document_corpus[name]
                )
                for name in selected_files
            )

            system_instruction = (
                "You are DocuMind Enterprise, a precise document assistant for "
                "business clients. Answer ONLY using the document context you "
                "were given below -- never use outside knowledge, and say "
                "clearly if the documents don't contain the answer. When you "
                "state a fact, name the specific file and page it came from "
                "(e.g. 'per contract.pdf, page 4'). Keep answers concise and "
                "direct. If asked to summarize, translate, or explain a "
                "section, do that fully and clearly."
            )

            st.session_state.gemini_chat = client.chats.create(
                model=MODEL_NAME,
                config=types.GenerateContentConfig(system_instruction=system_instruction),
                history=[
                    {"role": "user", "parts": [{"text": f"DOCUMENT CONTEXT:\n\n{compiled_corpus}"}]},
                    {"role": "model", "parts": [{"text": "Understood — I have the documents loaded and I'm ready for questions."}]},
                ],
            )
            st.session_state.primed_corpus_key = corpus_key
            st.session_state.chat_log = []
        except Exception as e:
            st.error(f"Failed to initialize the assistant: {e}")
            st.stop()

    for turn in st.session_state.chat_log:
        with st.chat_message(turn["role"]):
            st.markdown(
                f'<div class="response-container">{turn["text"]}</div>'
                if turn["role"] == "assistant" else turn["text"],
                unsafe_allow_html=True,
            )

    user_query = st.chat_input(
        "Ask anything (e.g., 'Summarize the whole document', "
        "'Translate the policy section into Spanish', 'Explain clause 4'):"
    )

    if user_query:
        st.session_state.chat_log.append({"role": "user", "text": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                try:
                    response = st.session_state.gemini_chat.send_message(user_query)
                    answer = response.text
                except Exception as e:
                    answer = f"Sorry, something went wrong generating a response: {e}"
            st.markdown(f'<div class="response-container">{answer}</div>', unsafe_allow_html=True)

        st.session_state.chat_log.append({"role": "assistant", "text": answer})
