# day66_streamlit_advanced/app.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(
    page_title="AI Assistant Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Base */
    .stApp { background-color: #f4f5f9; }
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* Header banner */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(102,126,234,0.25);
    }
    .app-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .app-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * { color: #e8e8f0 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

    [data-testid="stSidebar"] .stRadio > label { font-weight: 600; }
    [data-testid="stSidebar"] [role="radiogroup"] > label {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        transition: background 0.15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        background: rgba(255,255,255,0.14);
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .metric-card .label { font-size: 0.8rem; color: #6b7280; font-weight: 600; text-transform: uppercase; }
    .metric-card .value { font-size: 1.6rem; color: #1a1a2e; font-weight: 700; }

    [data-testid="stSidebar"] .metric-card {
        background: rgba(255,255,255,0.06);
        border-left: 4px solid #a78bfa;
    }
    [data-testid="stSidebar"] .metric-card .label { color: #c4c4d8 !important; }
    [data-testid="stSidebar"] .metric-card .value { color: white !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        width: 100%;
        padding: 0.55rem 0;
        font-weight: 600;
        transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.35);
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* Content cards for other pages */
    .content-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    /* Result badge */
    .result-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def metric_card(label: str, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_llm():
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=500
    )


llm = get_llm()

# Sidebar navigation
st.sidebar.markdown("## 🤖 AI Assistant Suite")
st.sidebar.caption("Day 66 · 100 Days of AI/ML")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["💬 Chat", "📝 Summarizer", "🏷️ Classifier", "ℹ️ About"],
    label_visibility="collapsed"
)
page = page.split(" ", 1)[1]  # strip emoji for logic below

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0

# Sidebar metrics
st.sidebar.markdown("---")
st.sidebar.markdown("**Session Stats**")
metric_card("Messages sent", st.session_state.total_messages)


# ── CHAT PAGE ────────────────────────────────────────────────
if page == "Chat":
    st.markdown("""
    <div class="app-header">
        <h1>💬 Chat</h1>
        <p>Multi-turn conversation with memory</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.info("Start typing below to begin a conversation with Claude.")

    for msg in st.session_state.chat_history:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type a message..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.total_messages += 1

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in st.session_state.chat_history
        ]

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = llm.invoke(messages)
            st.markdown(response.content)

        st.session_state.chat_history.append({"role": "assistant", "content": response.content})

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ── SUMMARIZER PAGE ──────────────────────────────────────────
elif page == "Summarizer":
    st.markdown("""
    <div class="app-header">
        <h1>📝 Text Summarizer</h1>
        <p>Paste any text and get a structured summary</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        text  = st.text_area("Enter text to summarize:", height=200, placeholder="Paste your text here...")
        style = st.selectbox("Summary style:", ["Bullet points", "One paragraph", "Key insights only"])
        summarize_clicked = st.button("Summarize")
        st.markdown('</div>', unsafe_allow_html=True)

    if summarize_clicked and text:
        style_prompt = {
            "Bullet points":     "Summarize in 5 bullet points using -> arrows.",
            "One paragraph":     "Summarize in exactly one concise paragraph.",
            "Key insights only": "Extract only the 3 most important insights.",
        }[style]

        with st.spinner("Summarizing..."):
            response = llm.invoke([HumanMessage(content=f"{style_prompt}\n\nText: {text}")])

        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### Summary")
        st.markdown(response.content)
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            metric_card("Input words", len(text.split()))
        with col2:
            metric_card("Style", style)


# ── CLASSIFIER PAGE ──────────────────────────────────────────
elif page == "Classifier":
    st.markdown("""
    <div class="app-header">
        <h1>🏷️ Text Classifier</h1>
        <p>Classify text into custom categories</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        text = st.text_input("Text to classify:", placeholder="Enter text here...")
        cats = st.text_input("Categories (comma separated):", value="positive, negative, neutral")
        classify_clicked = st.button("Classify")
        st.markdown('</div>', unsafe_allow_html=True)

    if classify_clicked and text and cats:
        categories = [c.strip() for c in cats.split(",")]
        prompt = f"Classify this text into exactly one of these categories: {categories}.\nReply with the category name only.\nText: {text}"

        with st.spinner("Classifying..."):
            response = llm.invoke([HumanMessage(content=prompt)])

        result = response.content.strip()

        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(f'Result: <span class="result-badge">{result}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            metric_card("Input length", f"{len(text)} chars")
        with col2:
            metric_card("Categories", len(categories))


# ── ABOUT PAGE ───────────────────────────────────────────────
elif page == "About":
    st.markdown("""
    <div class="app-header">
        <h1>ℹ️ About</h1>
        <p>AI Assistant Suite — Day 66 of 100 Days of AI/ML</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card">
    A multi-page Streamlit app demonstrating:

    - 💬 Multi-turn chat with conversation history
    - 📝 Text summarization with style options
    - 🏷️ Zero-shot text classification
    - 📊 Session state management across pages

    **Stack:** Streamlit + LangChain + Anthropic Claude
    </div>
    """, unsafe_allow_html=True)
