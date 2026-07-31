# day75_project3_start/agent.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from tools import search_product, get_product_reviews, compare_products, generate_buying_guide

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=2000
)

tools = [search_product, get_product_reviews, compare_products, generate_buying_guide]
agent = create_react_agent(llm, tools)


def research(query: str) -> str:
    result = agent.invoke({"messages": [("human", query)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("[PROJECT 3 — PRODUCT RESEARCH AGENT]\n")
    print("=" * 60)

    queries = [
        "I have a budget of Rs.25000 and I want the best headphones for music. What should I buy?",
        "Compare Sony WH-1000XM5 vs Apple AirPods Pro",
        "I need headphones under Rs.1500 for casual listening. What do you recommend?",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        print(f"\nAgent: {research(query)}")
        print("\n" + "=" * 60)