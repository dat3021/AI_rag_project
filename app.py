import sys
import os
from dotenv import load_dotenv
import streamlit as st
from router import router
from history_manager import HistoryManager

MODULE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "module")
sys.path.insert(0, MODULE_DIR)
load_dotenv()
# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.04);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #e2e8f0;
    }

    /* ── Header ── */
    .rag-header {
        text-align: center;
        padding: 2rem 0 1.5rem;
    }
    .rag-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }
    .rag-header p {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(8px);
        margin-bottom: 0.75rem;
        padding: 0.25rem 0.5rem;
    }

    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(96, 165, 250, 0.1) !important;
        border-color: rgba(96, 165, 250, 0.2) !important;
    }

    /* Assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: rgba(167, 139, 250, 0.08) !important;
        border-color: rgba(167, 139, 250, 0.15) !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: #e2e8f0 !important;
        font-size: 0.95rem;
        line-height: 1.65;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 16px !important;
        color: #e2e8f0 !important;
        backdrop-filter: blur(10px);
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(167, 139, 250, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.15) !important;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.75rem;
    }
    [data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #a78bfa !important;
        font-weight: 600;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4);
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(52, 211, 153, 0.12);
        border: 1px solid rgba(52, 211, 153, 0.25);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        color: #34d399;
        font-weight: 500;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background: #34d399;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0
if "history_manager" not in st.session_state:
    st.session_state.history_manager = HistoryManager()

# Build the router fresh on every run so hot-reloading works during development.
# router() returns a (execute, stream_execute) tuple.
router_execute, stream_router_execute = router(history_manager=st.session_state.history_manager)

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

_session_id = st.session_state.session_id

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 RAG Assistant")
    st.markdown(
        '<div class="status-badge"><span class="status-dot"></span>Online</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", st.session_state.total_questions)
    with col2:
        st.metric("Messages", len(st.session_state.messages))

    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    st.markdown(
        """
        <p style='color:#94a3b8; font-size:0.85rem;'>
        🧠 <b style='color:#e2e8f0'>LLM:</b> Gemini 2.5 Flash Lite<br>
        🔀 <b style='color:#e2e8f0'>Router:</b> History-Aware Gatekeeper<br>
        🗄️ <b style='color:#e2e8f0'>Vector DB:</b> ChromaDB<br>
        🔗 <b style='color:#e2e8f0'>Embeddings:</b> gemini-embedding-001<br>
        📄 <b style='color:#e2e8f0'>Source:</b> GitHub Markdown
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.total_questions = 0
        # Clear the history for this session in the HistoryManager
        st.session_state.history_manager.clear_history(_session_id)
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='color:#475569; font-size:0.75rem; text-align:center;'>Powered by LangChain + Gemini</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="rag-header">
        <h1>🤖 RAG Knowledge Assistant</h1>
        <p>Ask anything about your GitHub documentation repository</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Welcome message (shown only on a fresh session)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            """
            👋 **Hello! I'm your RAG-powered assistant.**

            I have access to your GitHub documentation and can answer questions about it in real time.

            **Try asking me:**
            - *"Who created the ELT Pipeline?"*
            - *"What is this project about?"*
            - *"Give me a summary of the main components."*
            """
        )

# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat Input & Streaming Response
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your documentation…", max_chars=2000):

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_questions += 1

    # Stream assistant response with typewriter effect
    with st.chat_message("assistant", avatar="🤖"):
        full_response = ""

        try:
            # st.write_stream consumes the generator and renders each chunk
            # as it arrives, producing the typewriter effect.
            # stream_router_execute handles: history check, query rewrite,
            # RAG streaming, and saving turns to HistoryManager internally.
            full_response = st.write_stream(
                stream_router_execute(prompt, session_id=_session_id)
            )

        except EnvironmentError as e:
            st.error(
                "⚠️ Configuration error: API key not found. "
                "Please check your `.env` file."
            )
            print(f"[RAG UI] EnvironmentError: {e}")
            full_response = ""

        except Exception:
            st.error(
                "⚠️ An unexpected error occurred while generating a response. "
                "Please try again."
            )
            print("[RAG UI] Unexpected error during stream_router_execute.")
            full_response = ""

    # st.write_stream already rendered the text; we just store it for history display
    if full_response:
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

