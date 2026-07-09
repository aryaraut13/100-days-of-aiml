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
    max_tokens=500
)


class State(TypedDict):
    messages: Annotated[List, operator.add]
    sentiment: str
    needs_escalation: bool
    response: str


def sentiment_node(state: State) -> dict:
    """Detect customer sentiment."""
    msg = state["messages"][-1].content
    response = llm.invoke([
        HumanMessage(content=f"Classify sentiment as 'positive', 'negative', or 'neutral'. Reply one word only.\nText: {msg}")
    ])
    sentiment = response.content.strip().lower()
    needs_escalation = "negative" in sentiment
    print(f"  [SENTIMENT] {sentiment} | escalate: {needs_escalation}")
    return {"sentiment": sentiment, "needs_escalation": needs_escalation}


def happy_response_node(state: State) -> dict:
    """Friendly response for positive/neutral customers."""
    msg = state["messages"][-1].content
    response = llm.invoke([HumanMessage(content=f"Give a friendly customer support response in 1 sentence: {msg}")])
    return {"response": f"[STANDARD] {response.content}"}


def escalation_node(state: State) -> dict:
    """Priority response for negative customers."""
    msg = state["messages"][-1].content
    response = llm.invoke([HumanMessage(content=f"Give an empathetic, priority customer support response in 1 sentence: {msg}")])
    return {"response": f"[ESCALATED] {response.content}"}


def route_by_sentiment(state: State) -> Literal["escalate", "standard"]:
    return "escalate" if state["needs_escalation"] else "standard"


builder = StateGraph(State)
builder.add_node("sentiment",  sentiment_node)
builder.add_node("standard",   happy_response_node)
builder.add_node("escalation", escalation_node)

builder.add_edge(START, "sentiment")
builder.add_conditional_edges(
    "sentiment",
    route_by_sentiment,
    {"escalate": "escalation", "standard": "standard"}
)
builder.add_edge("standard",   END)
builder.add_edge("escalation", END)

graph = builder.compile()


if __name__ == "__main__":
    print("[CUSTOMER SUPPORT ROUTING — CONDITIONAL BRANCHING]\n")

    messages = [
        "Thank you! My order arrived and it's perfect!",
        "This is absolutely terrible. My order is 2 weeks late and nobody responds.",
        "When will my order arrive?",
        "I've been waiting for 3 weeks. This is unacceptable. I want a refund NOW.",
    ]

    for msg in messages:
        print(f"Customer: {msg[:60]}...")
        result = graph.invoke({
            "messages": [HumanMessage(content=msg)],
            "sentiment": "",
            "needs_escalation": False,
            "response": ""
        })
        print(f"Response: {result['response'][:100]}...\n")