import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day47_project2_start'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'day48_project2_tools'))

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from tools import search_ecommerce_products, analyze_price_segments, get_market_trends
from search_tool import search_reviews
from report_tool import analyze_competitor, write_research_report

load_dotenv()

st.set_page_config(
    page_title="Market Research Agent",
    page_icon="🔍",
    layout="wide"
)

# Clean light theme CSS
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .main { background-color: #f8f9fa; }
    h1 { color: #1a1a2e; font-weight: 800; }
    h3 { color: #16213e; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
        transform: translateY(-1px);
    }
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .output-box {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    .badge {
        display: inline-block;
        background: #e8f4fd;
        color: #1a6fa8;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🔍 Ecommerce Market Research Agent")
st.markdown("**Autonomous market intelligence powered by Claude + LangChain Agents**")
st.markdown("""
<span class="badge">⚡ Claude Haiku</span>
<span class="badge">🦜 LangChain</span>
<span class="badge">🤖 ReAct Agent</span>
<span class="badge">6 Tools</span>
""", unsafe_allow_html=True)

st.divider()

@st.cache_resource
def get_agent():
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2000
    )
    tools = [
        search_ecommerce_products,
        analyze_price_segments,
        get_market_trends,
        search_reviews,
        analyze_competitor,
        write_research_report,
    ]
    return create_react_agent(llm, tools)

agent = get_agent()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Research Settings")

    category = st.selectbox(
        "Product Category",
        ["headphones", "earbuds", "laptop"],
    )

    research_type = st.radio(
        "Research Type",
        ["Full Market Report", "Competitor Analysis", "Custom Query"],
    )

    if research_type == "Custom Query":
        custom_query = st.text_area(
            "Enter your research query:",
            placeholder="e.g. What is the market opportunity for budget ANC headphones?",
            height=100
        )

    st.markdown("**Tools available:**")
    tools_list = [
        "🔎 Product Search",
        "📊 Price Segments",
        "📈 Market Trends",
        "💬 Review Analysis",
        "🏢 Competitor Analysis",
        "📝 Report Writing",
    ]
    for t in tools_list:
        st.markdown(f"<div class='metric-card'>{t}</div>", unsafe_allow_html=True)

    run_btn = st.button("🚀 Run Research", type="primary")

with col2:
    st.subheader("📋 Research Output")

    if run_btn:
        if research_type == "Full Market Report":
            task = f"""Research the {category} market completely:
            1. Search for top products
            2. Analyze price segments
            3. Get market trends and opportunities
            4. Write a comprehensive market report"""
        elif research_type == "Competitor Analysis":
            task = f"""Analyze the top competitors in {category}:
            1. Find top products and search reviews
            2. Analyze each as a competitor
            3. Write a competitive landscape report"""
        else:
            task = custom_query if custom_query else f"Research {category} market"

        with st.spinner(f"Agent researching {category} market... this may take 30-60 seconds"):
            result = agent.invoke({"messages": [("human", task)]})
            output = result["messages"][-1].content

        st.markdown(f"<div class='output-box'>{output}</div>", unsafe_allow_html=True)
        st.success("Research complete!")

    else:
        st.info("Configure your research settings and click **Run Research** to start.")

st.divider()
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.8rem;'>Built with LangChain Agents + Anthropic Claude | 100 Days of AI/ML — Day 49</p>",
    unsafe_allow_html=True
)