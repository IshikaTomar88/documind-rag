import streamlit as st
from pypdf import PdfReader
import io
import requests
import json
import time

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
st.markdown('<p class="sub-title">Upload client documents and chat naturally with full conversation history preserved.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Initialize Chat History in Session State
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------
# Sidebar: Credentials, New Chat Button & Document Vault
# ---------------------------------------------------------------------
with st.sidebar:
    # ChatGPT / Gemini Style "New Chat" Action Button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.header("🔑 Enterprise Configuration")
    
    # Pre-loaded with your specific key for seamless execution
    default_key = st.secrets.get("GEMINI_API_KEY", "AIzaSyCQx5EFZzndcE73GUWJchFG0OwkoToMsrM")
    api_key_input = st.text_input("Google Gemini API Key", value=default_key, type="password")
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

# ---------------------------------------------------------------------
# Main Execution Space
# ---------------------------------------------------------------------
if not uploaded_files:
    st.info("👈 Upload client documents in the sidebar to start chatting with your intelligence engine.")
    st.session_state.messages = []
else:
    if not api_key:
        st.warning("⚠️ Please provide a valid Google Gemini API key in the sidebar.")
        st.stop()

    # Consolidate text corpus cleanly with clear file boundaries
    compiled_corpus = "\n\n".join([f"=== DOCUMENT FILE: {name} ===\n{content}" for name, content in document_corpus.items()])
    
    # Display historical chat history bubbles so messages remain on screen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Box at the bottom
    user_query = st.chat_input("Ask anything about your documents (e.g., 'Is teamwork mentioned?', 'Summarize section 2'):")

    if user_query:
        # Save user message immediately to state and display
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant Response with Auto-Retry for 503 Server Spikes
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document corpus through Gemini 3.6 Flash..."):
                try:
                    system_instruction = (
                        "You are DocuMind Enterprise, an advanced document reasoning engine built for corporate clients. "
                        "Analyze the provided files with total precision. "
                        "1. If asked about exact terms or words, confirm their presence/absence and cite page numbers. "
                        "2. Provide clear, professional summaries, definitions, or translations as requested."
                    )

                    full_prompt = f"""
                    {system_instruction}

                    --- UPLOADED DOCUMENTS DATA ---
                    {compiled_corpus}
                    -------------------------------
                    """

                    # Build multi-turn context history
                    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
                    final_payload_text = f"{full_prompt}\n\nCONVERSATION HISTORY:\n{history_text}\n\nProvide the next professional response."

                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "contents": [{
                            "parts": [{"text": final_payload_text}]
                        }]
                    }

                    # Resilient 503 Auto-Retry Mechanism
                    max_retries = 3
                    success = False
                    output_text = ""
                    error_msg = ""

                    for attempt in range(max_retries):
                        response = requests.post(url, headers=headers, data=json.dumps(payload))
                        if response.status_code == 200:
                            res_json = response.json()
                            output_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                            success = True
                            break
                        elif response.status_code == 503:
                            time.sleep(2 * (attempt + 1))  # Wait briefly and retry automatically
                        else:
                            res_json = response.json()
                            error_msg = res_json.get("error", {}).get("message", f"API Error Code {response.status_code}")
                            break

                    if success:
                        st.markdown(output_text)
                        st.session_state.messages.append({"role": "assistant", "content": output_text})
                    else:
                        final_err = error_msg if error_msg else "Server is experiencing high traffic. Please retry in a moment."
                        st.error(final_err)

                except Exception as e:
                    st.error(f"Execution failed: {e}")
