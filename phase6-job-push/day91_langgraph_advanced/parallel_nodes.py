# day91_langgraph_advanced/parallel_nodes.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
import operator

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=200
)


class ResearchState(TypedDict):
    topic:           str
    market_data:     str
    competitor_data: str
    trend_data:      str
    final_report:    str


def market_node(state: ResearchState) -> dict:
    """Research market size and growth."""
    print(f"  [MARKET NODE] running...")
    response = llm.invoke([HumanMessage(
        content=f"Give 2 key market facts about {state['topic']} in India. Be brief."
    )])
    return {"market_data": response.content}


def competitor_node(state: ResearchState) -> dict:
    """Research competitors."""
    print(f"  [COMPETITOR NODE] running...")
    response = llm.invoke([HumanMessage(
        content=f"Name 3 key competitors in {state['topic']} market with one strength each. Be brief."
    )])
    return {"competitor_data": response.content}


def trend_node(state: ResearchState) -> dict:
    """Research trends."""
    print(f"  [TREND NODE] running...")
    response = llm.invoke([HumanMessage(
        content=f"List 2 key trends in {state['topic']} market. Be brief."
    )])
    return {"trend_data": response.content}


def synthesize_node(state: ResearchState) -> dict:
    """Combine all research into final report."""
    print(f"  [SYNTHESIZE NODE] combining all research...")
    combined = f"""
Market: {state['market_data']}
Competitors: {state['competitor_data']}
Trends: {state['trend_data']}
"""
    response = llm.invoke([HumanMessage(
        content=f"Write a 3-sentence executive summary from this research:\n{combined}"
    )])
    return {"final_report": response.content}


# Build graph with parallel branches
builder = StateGraph(ResearchState)

builder.add_node("market",     market_node)
builder.add_node("competitor", competitor_node)
builder.add_node("trend",      trend_node)
builder.add_node("synthesize", synthesize_node)

# All three research nodes run in parallel from START
builder.add_edge(START,        "market")
builder.add_edge(START,        "competitor")
builder.add_edge(START,        "trend")

# All three converge into synthesize
builder.add_edge("market",     "synthesize")
builder.add_edge("competitor", "synthesize")
builder.add_edge("trend",      "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()


if __name__ == "__main__":
    print("[LANGGRAPH — PARALLEL NODES]\n")
    print("Graph: START -> [market | competitor | trend] -> synthesize -> END")
    print("All 3 research nodes run simultaneously\n")

    start = time.time()
    result = graph.invoke({"topic": "wireless earbuds", "market_data": "", "competitor_data": "", "trend_data": "", "final_report": ""})
    elapsed = time.time() - start

    print(f"\n[FINAL REPORT]")
    print(result["final_report"])
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"(Sequential would take ~3x longer)")