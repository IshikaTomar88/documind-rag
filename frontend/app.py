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
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">💼 DocuMind Enterprise AI Workspace</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload client documents and leverage instant executive intelligence.</p>', unsafe_allow_html=True)

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Credentials, New Chat Button & Document Vault
with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.header("🔑 Enterprise Configuration")
    
    # Securely pulls from Streamlit Secrets or leaves blank for safe manual input (NO hardcoded keys)
    default_key = st.secrets.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input("Google Gemini API Key", value=default_key, type="password", help="Paste your new Google AI Studio API key here.")
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

# Main Execution Space
if not uploaded_files:
    st.info("👈 Upload client documents in the sidebar to activate the DocuMind intelligence engine.")
    st.session_state.messages = []
else:
    if not api_key:
        st.warning("⚠️ Please provide your Google Gemini API key in the sidebar to proceed.")
        st.stop()

    compiled_corpus = "\n\n".join([f"=== DOCUMENT FILE: {name} ===\n{content}" for name, content in document_corpus.items()])
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown("##### ⚡ Executive Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    triggered_quick_prompt = None
    with col1:
        if st.button("📝 Executive Summary", use_container_width=True):
            triggered_quick_prompt = "Provide a comprehensive executive summary of the uploaded documents, highlighting core objectives and key takeaways."
    with col2:
        if st.button("⚠️ Risk & Liability Analysis", use_container_width=True):
            triggered_quick_prompt = "Analyze the documents for potential risks, liabilities, or critical clauses. List them clearly with recommendations."
    with col3:
        if st.button("📅 Dates & Action Items", use_container_width=True):
            triggered_quick_prompt = "Extract all critical deadlines, dates, financial figures, and action items mentioned in the documents."

    user_query = st.chat_input("Ask anything about your documents...")
    active_prompt = user_query if user_query else triggered_quick_prompt

    if active_prompt:
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document corpus through Gemini 3.6 Flash..."):
                try:
                    # Initialize official Google GenAI client securely
                    client = genai.Client(api_key=api_key)
                    
                    system_instruction = (
                        "You are DocuMind Enterprise, an advanced document reasoning engine built for corporate clients. "
                        "Analyze the provided files with total precision. "
                        "1. If asked about exact terms or words, confirm their presence/absence and cite page numbers. "
                        "2. Provide clear, professional summaries, risk assessments, definitions, or translations as requested."
                    )

                    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages[:-1]])
                    
                    full_prompt = f"""
                    {system_instruction}

                    --- UPLOADED DOCUMENTS DATA ---
                    {compiled_corpus}
                    
                    CONVERSATION HISTORY:
                    {history_text}
                    -------------------------------
                    
                    CURRENT USER REQUEST: {active_prompt}
                    """

                    response = client.models.generate_content(
                        model='gemini-3.6-flash',
                        contents=full_prompt
                    )

                    output_text = response.text
                    st.markdown(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                    
                    if triggered_quick_prompt:
                        st.rerun()

                except Exception as e:
                    st.error(f"Execution failed: {e}")
