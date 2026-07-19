import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from tools import search_ecommerce_products, analyze_price_segments, get_market_trends

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=2000
)

tools = [search_ecommerce_products, analyze_price_segments, get_market_trends]
agent = create_react_agent(llm, tools)


def research(category: str) -> str:
    task = f"""Research the {category} market completely:
    1. Search for top products in this category
    2. Analyze the price segments
    3. Get market trends and opportunities
    Then provide a comprehensive market overview."""

    result = agent.invoke({"messages": [("human", task)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("[PROJECT 2 — MARKET RESEARCH AGENT]\n")
    print("Researching: headphones market\n")
    report = research("headphones")
    print(report)