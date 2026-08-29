import streamlit as st
import os
import io

st.set_page_config(
    page_title="DocuMind - AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { color: #64748B; font-size: 1.1rem; margin-bottom: 2rem; }
    .citation-box { background-color: #F8FAFC; border-left: 4px solid #3B82F6; padding: 15px; border-radius: 4px; margin: 15px 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📄 DocuMind</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ask questions about your own documents. Every answer is grounded in — and cited back to — the exact source excerpt it came from.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar: Document Vault
# ---------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/document.png", width=60)
    st.header("📁 Document Vault")
    
    uploaded_files = st.file_uploader(
        "Upload your text/data files",
        type=["txt", "csv", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"Successfully loaded {len(uploaded_files)} file(s).")
        st.markdown("### Active Files")
        for file in uploaded_files:
            st.text(f"• {file.name}")

# ---------------------------------------------------------------------
# Main Workspace
# ---------------------------------------------------------------------
if not uploaded_files:
    st.info("👈 Upload documents in the sidebar to activate DocuMind's search and citation engine.")
    
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        1. Upload one or more text (`.txt`), CSV (`.csv`), or markdown (`.md`) files using the sidebar.
        2. Type any question related to your document contents in the chat box.
        3. DocuMind instantly scans the text chunks, retrieves the best matches, and cites the exact source file and excerpt!
        """)
else:
    # Read and chunk documents into memory
    document_corpus = {}
    for file in uploaded_files:
        try:
            content = file.read().decode("utf-8", errors="ignore")
            # Split document into paragraph chunks for precise citation tracking
            chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
            document_corpus[file.name] = chunks if chunks else [content]
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")

    # Query Input
    query = st.text_input("💬 Ask a question about your documents:", placeholder="e.g., What are the core project milestones or key figures mentioned?")

    if query:
        with st.spinner("Searching document corpus and matching excerpts..."):
            matching_results = []
            query_terms = set(query.lower().split())

            # Search across all uploaded files and chunks
            for filename, chunks in document_corpus.items():
                for idx, chunk in enumerate(chunks):
                    chunk_lower = chunk.lower()
                    # Calculate keyword match score
                    score = sum(1 for term in query_terms if term in chunk_lower)
                    if score > 0:
                        matching_results.append({
                            "file": filename,
                            "chunk_index": idx,
                            "score": score,
                            "text": chunk
                        })

            # Sort results by highest match score
            matching_results = sorted(matching_results, key=lambda x: x["score"], reverse=True)

        st.markdown("### 💡 Grounded Answer & Citations")

        if not matching_results:
            st.warning("No direct matches found for your query. Try using different keywords or check if the information is inside the uploaded files.")
        else:
            # Display primary best match with strict citation block
            best = matching_results[0]
            st.markdown(f"""
                <div class="citation-box">
                    <p style="font-size: 1.05rem; font-weight: 500; color: #1E293B;">"{best['text']}"</p>
                    <hr style="margin: 8px 0; border-color: #E2E8F0;">
                    <p style="font-size: 0.85rem; color: #64748B; margin-bottom: 0;">
                        📌 <b>Source Citation:</b> File: <code>{best['file']}</code> (Relevance Score: {best['score']})
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Display secondary matching excerpts if available
            if len(matching_results) > 1:
                with st.expander(f"📚 View {len(matching_results) - 1} additional matching excerpt(s)"):
                    for alt in matching_results[1:5]:
                        st.markdown(f"**File:** `{alt['file']}`")
                        st.markdown(f"> {alt['text']}")
                        st.divider()
