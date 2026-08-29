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
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">💼 DocuMind Enterprise AI Workspace</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload business contracts, technical manuals, HR policies, or messy PDFs. Get precise answers, exact word checks, translations, and executive summaries instantly.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar: Credentials & Secure Document Vault
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Enterprise Configuration")
    
    # Check Streamlit secrets first, otherwise provide input field
    default_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
    api_key_input = st.text_input("Google Gemini API Key", value=default_key, type="password", help="Enter your Gemini API key here.")
    
    # Clean up any accidental spaces around the key
    api_key = api_key_input.strip() if api_key_input else ""
    
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
                
                # Lightweight page-by-page PDF extraction
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
    st.info("👈 Enter your Gemini API key in the sidebar and upload your client documents to activate the engine.")
else:
    # Explicit validation check to prevent blank key errors
    if not api_key:
        st.warning("⚠️ Please paste your Google Gemini API key into the sidebar text box above to enable AI processing.")
        st.stop()

    # Consolidate text corpus cleanly with clear file boundaries
    compiled_corpus = "\n\n".join([f"=== DOCUMENT FILE: {name} ===\n{content}" for name, content in document_corpus.items()])
    
    # Universal Client Prompt Bar
    user_query = st.chat_input("Ask anything about your documents (e.g., 'Is the word teamwork present?', 'Summarize the agreement', 'Translate section 1 into French'):")

    if user_query:
        with st.spinner("Processing document corpus through Gemini intelligence core..."):
            try:
                # Initialize official Google GenAI client with validated key
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

                # Call Gemini 2.5 Flash model
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )

                st.markdown("### 💡 Professional Analysis Output")
                st.markdown(f'<div class="response-container">{response.text}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Execution failed: {e}")
