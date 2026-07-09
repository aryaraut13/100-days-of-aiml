import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
import operator

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=300
)


class State(TypedDict):
    messages: Annotated[List, operator.add]
    query_type: str
    response: str


def classify_node(state: State) -> dict:
    """Classify the query as technical or general."""
    query = state["messages"][-1].content
    response = llm.invoke([
        HumanMessage(content=f"""Classify this query as exactly one word: 'technical' or 'general'.
Query: {query}
Reply with only the word.""")
    ])
    query_type = response.content.strip().lower()
    if "technical" in query_type:
        query_type = "technical"
    else:
        query_type = "general"

    print(f"  [CLASSIFY] '{query[:40]}...' -> {query_type}")
    return {"query_type": query_type}


def technical_node(state: State) -> dict:
    """Handle technical queries with detailed explanation."""
    print(f"  [TECHNICAL NODE] handling query")
    query = state["messages"][-1].content
    response = llm.invoke([
        HumanMessage(content=f"Give a technical explanation in 2 sentences: {query}")
    ])
    return {
        "messages": [AIMessage(content=response.content)],
        "response": f"[TECHNICAL] {response.content}"
    }


def general_node(state: State) -> dict:
    """Handle general queries with simple explanation."""
    print(f"  [GENERAL NODE] handling query")
    query = state["messages"][-1].content
    response = llm.invoke([
        HumanMessage(content=f"Give a simple, friendly explanation in 2 sentences: {query}")
    ])
    return {
        "messages": [AIMessage(content=response.content)],
        "response": f"[GENERAL] {response.content}"
    }


def route_query(state: State) -> Literal["technical", "general"]:
    """Routing function — decides which node to go to."""
    return state["query_type"]


# Build graph with conditional edges
builder = StateGraph(State)
builder.add_node("classify",  classify_node)
builder.add_node("technical", technical_node)
builder.add_node("general",   general_node)

builder.add_edge(START, "classify")

# Conditional edge — routes based on query_type
builder.add_conditional_edges(
    "classify",
    route_query,
    {
        "technical": "technical",
        "general":   "general",
    }
)

builder.add_edge("technical", END)
builder.add_edge("general",   END)

graph = builder.compile()


if __name__ == "__main__":
    print("[LANGGRAPH — CONDITIONAL EDGES]\n")
    print("Flow: START -> classify -> [technical OR general] -> END\n")

    queries = [
        "What is gradient descent in machine learning?",
        "What should I have for breakfast?",
        "How does a transformer architecture work?",
        "What is the capital of France?",
    ]

    for query in queries:
        print(f"Query: {query}")
        result = graph.invoke({
            "messages": [HumanMessage(content=query)],
            "query_type": "",
            "response": ""
        })
        print(f"Response: {result['response'][:100]}...\n")