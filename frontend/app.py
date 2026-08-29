import streamlit as st
from pypdf import PdfReader
import io
from google import genai

# Page Configuration
st.set_page_config(
    page_title="DocuMind Enterprise - Document Intelligence Engine",
    page_icon="💼",
    layout="wide"
)

# Professional Business Styling
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin-bottom: 0px; }
    .sub-title { color: #475569; font-size: 1.05rem; margin-bottom: 2rem; }
    .response-container { background-color: #F8FAFC; border: 1px solid #CBD5E1; padding: 24px; border-radius: 10px; margin-top: 15px; }
    .metric-card { background-color: #F1F5F9; padding: 15px; border-radius: 8px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">💼 DocuMind Enterprise AI Workspace</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload business contracts, technical manuals, HR policies, or messy PDFs. Get precise answers, exact word checks, translations, and executive summaries instantly.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar: Credentials & Secure Document Vault
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Enterprise Configuration")
    
    # Secure API key input (or reads from Streamlit secrets if pre-configured)
    default_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
    api_key = st.text_input("Google Gemini API Key", value=default_key, type="password", help="Enter your free Gemini API key from Google AI Studio.")
    
    st.markdown("[Get a free Gemini API key here](https://aistudio.google.com/apikey)")
    
    st.divider()
    st.header("📁 Client Document Vault")
    
    uploaded_files = st.file_uploader(
        "Upload multi-page PDFs or text files",
        type=["pdf", "txt", "csv", "md"],
        accept_multiple_files=True
    )

    document_corpus = {}

    if uploaded_files:
        for file in uploaded_files:
            try:
                file_bytes = file.read()
                file_text = ""
                
                # Lightweight page-by-page PDF extraction to preserve low memory usage
                if file.name.lower().endswith(".pdf"):
                    reader = PdfReader(io.BytesIO(file_bytes))
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            file_text += f"\n[Page {i+1}]\n" + page_text
                else:
                    file_text = file_bytes.decode("utf-8", errors="ignore")
                
                document_corpus[file.name] = file_text
                st.success(f"Indexed: {file.name} ({len(file_text.split()):,} words)")
            except Exception as e:
                st.error(f"Error parsing {file.name}: {e}")

# ---------------------------------------------------------------------
# Main Execution Space
# ---------------------------------------------------------------------
if not uploaded_files:
    st.info("👈 Enter your Gemini API key and upload your client documents in the sidebar to activate the enterprise engine.")
    
    with st.expander("🏢 Real-World Business Capabilities"):
        st.markdown("""
        * **Exact Term Verification:** Ask things like *"Is the word 'teamwork' or 'liability' present in the PDF?"* and get exact positional tracking.
        * **Deep Document Summaries:** Instantly synthesize multi-page manuals or legal handbooks into clean executive summaries.
        * **Multi-Language Translation:** Translate any clause or section seamlessly into other languages.
        * **Complex Q&A:** Handle open-ended business reasoning questions just like ChatGPT or Gemini.
        """)
else:
    if not api_key:
        st.warning("⚠️ Please provide your Google Gemini API key in the sidebar to enable the intelligence core.")
        st.stop()

    # Consolidate text corpus cleanly with clear file boundaries for the cloud model
    compiled_corpus = "\n\n".join([f"=== DOCUMENT FILE: {name} ===\n{content}" for name, content in document_corpus.items()])
    
    # Universal Client Prompt Bar
    user_query = st.chat_input("Ask anything about your documents (e.g., 'Is the word teamwork present?', 'Summarize the agreement', 'Translate section 1 into French'):")

    if user_query:
        with st.spinner("Processing document corpus through Gemini 3.7 Flash engine..."):
            try:
                # Initialize official Google GenAI client (computationally lightweight locally)
                client = genai.Client(api_key=api_key)
                
                system_instruction = (
                    "You are DocuMind Enterprise, an advanced document reasoning engine built for corporate clients. "
                    "Your objective is to analyze the provided files with total precision. "
                    "1. If the user asks if a specific word, phrase, or term (like 'teamwork') is present, search the texts, confirm its exact presence or absence, and cite the relevant pages/files. "
                    "2. If they request summaries, structural breakdowns, definitions, or translations, execute them comprehensively and professionally."
                )

                prompt = f"""
                {system_instruction}

                --- UPLOADED DOCUMENTS DATA ---
                {compiled_corpus}
                -------------------------------

                CLIENT REQUEST: {user_query}
                """

                # Call Gemini 3.7 Flash for deep context reasoning
                response = client.models.generate_content(
                    model='gemini-3.7-flash',
                    contents=prompt
                )

                st.markdown("### 💡 Professional Analysis Output")
                st.markdown(f'<div class="response-container">{response.text}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Execution failed: {e}")
