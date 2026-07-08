"""
Day 51 - LangGraph Basics: State Machines and Graphs
Building a simple 3-node state graph to understand LangGraph fundamentals
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
import time


class AgentState(TypedDict):
    input_text: str
    word_count: int
    char_count: int
    steps_completed: int
    log: list


def node_analyze(state: AgentState) -> AgentState:
    text = state["input_text"]
    state["word_count"] = len(text.split())
    state["char_count"] = len(text)
    state["steps_completed"] += 1
    state["log"].append(f"Step {state['steps_completed']}: analyze -> words={state['word_count']}, chars={state['char_count']}")
    return state


def node_transform(state: AgentState) -> AgentState:
    state["input_text"] = state["input_text"].upper()
    state["steps_completed"] += 1
    state["log"].append(f"Step {state['steps_completed']}: transform -> text upper-cased")
    return state


def node_summarize(state: AgentState) -> AgentState:
    state["steps_completed"] += 1
    state["log"].append(f"Step {state['steps_completed']}: summarize -> done")
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("analyze", node_analyze)
    graph.add_node("transform", node_transform)
    graph.add_node("summarize", node_summarize)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "transform")
    graph.add_edge("transform", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


if __name__ == "__main__":
    start_time = time.time()

    app = build_graph()

    initial_state: AgentState = {
        "input_text": "LangGraph makes building AI agent workflows structured and debuggable",
        "word_count": 0,
        "char_count": 0,
        "steps_completed": 0,
        "log": []
    }

    print("=" * 60)
    print("DAY 51 - LANGGRAPH BASICS - STATE MACHINE EXECUTION")
    print("=" * 60)

    final_state = app.invoke(initial_state)

    print("\n--- EXECUTION LOG ---")
    for entry in final_state["log"]:
        print(entry)

    print("\n--- FINAL STATE ---")
    print(f"Original word count : {final_state['word_count']}")
    print(f"Original char count : {final_state['char_count']}")
    print(f"Steps completed     : {final_state['steps_completed']}")
    print(f"Final text          : {final_state['input_text']}")

    elapsed = round(time.time() - start_time, 4)
    print(f"\nTotal execution time: {elapsed}s")
    print("=" * 60)