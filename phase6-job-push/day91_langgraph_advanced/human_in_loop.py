# day91_langgraph_advanced/human_in_loop.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import operator

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=300
)


class ActionState(TypedDict):
    task:          str
    plan:          str
    approved:      bool
    result:        str
    messages:      Annotated[List, operator.add]


def plan_node(state: ActionState) -> dict:
    """Generate a plan for the task."""
    response = llm.invoke([HumanMessage(
        content=f"Create a 3-step action plan for: {state['task']}. Number each step."
    )])
    print(f"\n[PLAN GENERATED]\n{response.content}")
    return {"plan": response.content}


def human_review_node(state: ActionState) -> dict:
    """Pause for human approval."""
    print(f"\n[HUMAN REVIEW REQUIRED]")
    print(f"Plan: {state['plan'][:100]}...")
    print(f"Type 'yes' to approve or 'no' to reject: ", end="")
    answer = input().strip().lower()
    approved = answer == "yes"
    print(f"Decision: {'APPROVED' if approved else 'REJECTED'}")
    return {"approved": approved}


def execute_node(state: ActionState) -> dict:
    """Execute the approved plan."""
    response = llm.invoke([HumanMessage(
        content=f"Execute this plan and summarize the outcome in 2 sentences:\n{state['plan']}"
    )])
    return {"result": f"EXECUTED: {response.content}"}


def reject_node(state: ActionState) -> dict:
    """Handle rejected plans."""
    return {"result": "Plan rejected by human reviewer. Task cancelled."}


def route_after_review(state: ActionState) -> str:
    return "execute" if state["approved"] else "reject"


builder  = StateGraph(ActionState)
memory   = MemorySaver()

builder.add_node("plan",         plan_node)
builder.add_node("human_review", human_review_node)
builder.add_node("execute",      execute_node)
builder.add_node("reject",       reject_node)

builder.add_edge(START,          "plan")
builder.add_edge("plan",         "human_review")
builder.add_conditional_edges(
    "human_review",
    route_after_review,
    {"execute": "execute", "reject": "reject"}
)
builder.add_edge("execute",      END)
builder.add_edge("reject",       END)

graph = builder.compile(checkpointer=memory)


if __name__ == "__main__":
    print("[HUMAN-IN-THE-LOOP AGENT]\n")
    print("The agent will generate a plan and ask for your approval.\n")

    config = {"configurable": {"thread_id": "review_session_1"}}
    result = graph.invoke(
        {
            "task":     "Send promotional emails to 10,000 customers about a 50% discount",
            "plan":     "",
            "approved": False,
            "result":   "",
            "messages": []
        },
        config=config
    )
    print(f"\n[OUTCOME]\n{result['result']}")