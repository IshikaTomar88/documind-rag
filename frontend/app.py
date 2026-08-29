import streamlit as st
from pypdf import PdfReader
import io
from google import genai

st.set_page_config(
    page_title="DocuMind - AI PDF & Document Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 2.3rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { color: #64748B; font-size: 1.05rem; margin-bottom: 2rem; }
    .response-box { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 20px; border-radius: 8px; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🤖 DocuMind AI Workspace</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload PDFs or text files and chat with them using Google Gemini AI. Ask anything: summaries, keyword checks, or translations.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar: Settings & Document Vault
# ---------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/pdf-2.png", width=60)
    st.header("🔑 AI Configuration")
    
    # Secure API key input (or checks Streamlit secrets)
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key to activate advanced reasoning.")
    
    st.divider()
    st.header("📁 Document Vault")
    uploaded_files = st.file_uploader(
        "Upload PDF or Text files",
        type=["pdf", "txt", "csv", "md"],
        accept_multiple_files=True
    )

    document_corpus = {}

    if uploaded_files:
        for file in uploaded_files:
            try:
                file_bytes = file.read()
                file_text = ""
                
                if file.name.lower().endswith(".pdf"):
                    reader = PdfReader(io.BytesIO(file_bytes))
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            file_text += f"\n--- Page {i+1} ---\n" + text
                else:
                    file_text = file_bytes.decode("utf-8", errors="ignore")
                
                document_corpus[file.name] = file_text
                st.success(f"Loaded: {file.name}")
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")

# ---------------------------------------------------------------------
# Main Assistant Interface
# ---------------------------------------------------------------------
if not uploaded_files:
    st.info("👈 Please enter your API key and upload one or more PDF files in the sidebar to begin.")
else:
    if not api_key:
        st.warning("⚠️ Please provide your Google Gemini API key in the sidebar to enable AI responses.")
        st.stop()

    # Compile the full text of all uploaded documents
    full_corpus_text = "\n\n".join([f"=== DOCUMENT: {name} ===\n{content}" for name, content in document_corpus.items()])
    
    # Chat Input for any type of question
    user_query = st.chat_input("Ask anything (e.g., 'Is the word teamwork present?', 'Summarize the whole PDF', 'Translate section 1 into Spanish'):")

    if user_query:
        with st.spinner("Gemini AI is analyzing your documents..."):
            try:
                # Initialize Gemini Client using official SDK
                client = genai.Client(api_key=api_key)
                
                # Construct prompt context instructions
                system_instruction = (
                    "You are DocuMind, an advanced AI document assistant like Gemini/ChatGPT. "
                    "Your job is to answer the user's questions based strictly or contextually on the provided documents. "
                    "If the user asks if a specific word (like 'teamwork') is present, search the document text, verify if it is there, "
                    "and state how many times or where it appears. If they ask for a summary, translation, or definition, fulfill it completely."
                )
                
                prompt = f"""
                {system_instruction}

                --- DOCUMENTS CONTENT ---
                {full_corpus_text}
                -----------------------

                USER QUESTION: {user_query}
                """

                # Call Gemini model
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )

                st.markdown("### 💡 AI Response")
                st.markdown(f'<div class="response-box">{response.text}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
