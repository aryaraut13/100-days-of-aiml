# day76_project3_build/app.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day75_project3_start'))

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from tools import search_product, get_product_reviews, compare_products, generate_buying_guide

load_dotenv()

st.set_page_config(
    page_title="Product Research Agent",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #f4f5f9; }
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* Header banner */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 18px rgba(102,126,234,0.28);
    }
    .app-header h1 {
        color: white;
        margin: 0;
        font-size: 1.9rem;
        font-weight: 800;
    }
    .app-header p {
        color: rgba(255,255,255,0.88);
        margin: 0.35rem 0 0.9rem 0;
        font-size: 0.98rem;
    }

    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        font-weight: 600;
    }

    /* Settings panel */
    .settings-card {
        background: white;
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .settings-card h3 {
        margin-top: 0;
        font-size: 1.05rem;
        color: #1a1a2e;
    }

    /* Result panel */
    .result-box {
        background: white;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .result-box-header {
        font-weight: 700;
        color: #1a1a2e;
        font-size: 1rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .empty-state {
        background: white;
        border-radius: 14px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        color: #6b7280;
    }
    .empty-state .icon { font-size: 2.2rem; margin-bottom: 0.6rem; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        width: 100%;
        padding: 0.6rem 0;
        font-weight: 700;
        transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(102,126,234,0.35);
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>🛍️ AI Product Research Agent</h1>
    <p>Get instant buying recommendations powered by Claude + LangChain</p>
    <span class="badge">🔍 Product Search</span>
    <span class="badge">⭐ Review Analysis</span>
    <span class="badge">📊 Comparison</span>
    <span class="badge">🎯 Buying Guide</span>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_agent():
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2000
    )
    tools = [search_product, get_product_reviews, compare_products, generate_buying_guide]
    return create_react_agent(llm, tools)


agent = get_agent()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="settings-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Research Settings")

    query_type = st.radio(
        "What do you need?",
        ["Find best product for my budget",
         "Compare two products",
         "Get product details",
         "Custom query"]
    )

    if query_type == "Find best product for my budget":
        budget   = st.number_input("Budget (Rs.)", min_value=500, max_value=100000, value=5000, step=500)
        use_case = st.selectbox("Use case", ["music", "calls", "casual", "exercise", "gaming"])
        category = st.selectbox("Category", ["headphones", "earbuds"])
        query    = f"I have a budget of Rs.{budget} and need {category} for {use_case}. What should I buy? Search for options and give me a recommendation with reasons."

    elif query_type == "Compare two products":
        p1    = st.text_input("Product 1", value="Sony WH-1000XM5")
        p2    = st.text_input("Product 2", value="Apple AirPods Pro")
        query = f"Compare {p1} vs {p2}. Include pricing, ratings, pros and cons, and a final recommendation."

    elif query_type == "Get product details":
        product = st.text_input("Product name", value="boAt Rockerz 450")
        query   = f"Give me complete details about {product} including price, specs, reviews, and whether I should buy it."

    else:
        query = st.text_area("Enter your research query:", height=100,
                              placeholder="e.g. What are the best budget headphones under Rs.2000?")

    run_btn = st.button("🚀 Research Now", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if run_btn and query:
        with st.spinner("Agent researching... this may take 20-30 seconds"):
            result = agent.invoke({"messages": [("human", query)]})
            output = result["messages"][-1].content

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<div class="result-box-header">📋 Research Result</div>', unsafe_allow_html=True)
        st.markdown(output)  # rendered as real markdown, not raw HTML text
        st.markdown('</div>', unsafe_allow_html=True)

        st.success("Research complete!")

        with st.expander("🔍 View query sent to agent"):
            st.write(query)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">🛍️</div>
            <div>Configure your research on the left and click <b>Research Now</b>.</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.markdown(
    "<p style='text-align:center;color:#888;font-size:0.8rem;'>Built with LangChain Agents + Claude | 100 Days of AI/ML Day 76</p>",
    unsafe_allow_html=True
)