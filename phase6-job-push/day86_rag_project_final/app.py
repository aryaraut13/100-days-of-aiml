import streamlit as st
from rag_bot import build_rag_chain, ask

st.set_page_config(
    page_title="ShopAssist AI",
    page_icon="🛒",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0f0f0f;
    }
    .stApp {
        background-color: #0f0f0f;
    }
    .header-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header-subtitle {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .badge {
        background: #1a1a2e;
        border: 1px solid #333;
        color: #888;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
    }
    .stChatMessage {
        background-color: #1a1a1a !important;
        border-radius: 12px !important;
        margin: 0.5rem 0 !important;
    }
    .stChatInputContainer {
        border-top: 1px solid #333 !important;
        padding-top: 1rem !important;
    }
    div[data-testid="stChatInput"] {
        background-color: #1a1a1a !important;
        border: 1px solid #333 !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <p class="header-title">🛒 ShopAssist AI</p>
    <p class="header-subtitle">Instant answers from your store knowledge base</p>
</div>
<div class="badge-container">
    <span class="badge">⚡ Claude Haiku</span>
    <span class="badge">🔍 ChromaDB</span>
    <span class="badge">🦜 LangChain</span>
    <span class="badge">🤗 HuggingFace Embeddings</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# Suggested questions
st.markdown("**💡 Try asking:**")
cols = st.columns(2)
suggestions = [
    "How do I return a product?",
    "What payment methods do you accept?",
    "Do you offer EMI options?",
    "How do I track my order?",
]
for i, suggestion in enumerate(suggestions):
    with cols[i % 2]:
        if st.button(suggestion, use_container_width=True, key=f"btn_{i}"):
            st.session_state.pending_question = suggestion

st.divider()

# Build chain
@st.cache_resource
def get_chain():
    return build_rag_chain()

chain = get_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I'm ShopAssist AI. Ask me anything about returns, shipping, payments, or orders. I'll answer based on our store's knowledge base."
    })

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle suggested question click
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            response = ask(chain, question)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about returns, shipping, payments..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            response = ask(chain, prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.75rem;'>Built with LangChain + ChromaDB + Anthropic Claude | RAG Pipeline</p>",
    unsafe_allow_html=True
)