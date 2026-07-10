import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
import operator
from workers import run_research_agent, run_writing_agent

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=500
)


class SupervisorState(TypedDict):
    messages: Annotated[List, operator.add]
    task: str
    research_output: str
    writing_output: str
    next_worker: str
    final_report: str
    iteration: int


def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor decides what to do next."""
    iteration = state.get("iteration", 0)
    print(f"  [SUPERVISOR] iteration {iteration}")

    if iteration == 0:
        print(f"  [SUPERVISOR] -> delegating to RESEARCHER")
        return {"next_worker": "researcher", "iteration": iteration + 1}
    elif iteration == 1:
        print(f"  [SUPERVISOR] -> delegating to WRITER")
        return {"next_worker": "writer", "iteration": iteration + 1}
    else:
        print(f"  [SUPERVISOR] -> task complete")
        return {"next_worker": "FINISH", "iteration": iteration + 1}


def researcher_node(state: SupervisorState) -> dict:
    """Research worker executes research task."""
    print(f"  [RESEARCHER] working...")
    task    = state["task"]
    result  = run_research_agent(f"Research this topic and gather key data: {task}")
    print(f"  [RESEARCHER] done")
    return {"research_output": result}


def writer_node(state: SupervisorState) -> dict:
    """Writing worker creates the report."""
    print(f"  [WRITER] working...")
    research = state["research_output"]
    task     = state["task"]
    result   = run_writing_agent(
        f"Write a structured market report about '{task}' using this research:\n{research}"
    )
    print(f"  [WRITER] done")
    return {"writing_output": result, "final_report": result}


def route_supervisor(state: SupervisorState) -> Literal["researcher", "writer", "FINISH"]:
    return state["next_worker"]


# Build supervisor graph
builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer",     writer_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "writer":     "writer",
        "FINISH":     END,
    }
)
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer",     "supervisor")

graph = builder.compile()


if __name__ == "__main__":
    print("[MULTI-AGENT SYSTEM — SUPERVISOR + WORKERS]\n")
    print("Architecture: Supervisor -> Researcher -> Supervisor -> Writer -> END\n")

    task = "headphones market analysis focusing on boAt and Sony"
    print(f"Task: {task}\n")

    result = graph.invoke({
        "messages":        [HumanMessage(content=task)],
        "task":            task,
        "research_output": "",
        "writing_output":  "",
        "next_worker":     "",
        "final_report":    "",
        "iteration":       0,
    })

    print("\n[FINAL REPORT]")
    print(result["final_report"][:500] + "...")