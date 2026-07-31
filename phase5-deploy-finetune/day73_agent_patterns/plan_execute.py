# day73_agent_patterns/plan_execute.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1000
)


def plan(task: str) -> list:
    """Step 1: Create a plan before executing."""
    response = llm.invoke([HumanMessage(content=f"""
You are a planning agent. Break this task into 3-5 concrete steps.
Return ONLY a numbered list. No explanations.

Task: {task}
""")])
    lines = [l.strip() for l in response.content.strip().split('\n') if l.strip()]
    steps = [l for l in lines if l[0].isdigit()]
    return steps


def execute_step(step: str, context: str) -> str:
    """Step 2: Execute each step with accumulated context."""
    response = llm.invoke([HumanMessage(content=f"""
Execute this step concisely (2-3 sentences max).
Previous context: {context[:200] if context else 'None'}
Step to execute: {step}
""")])
    return response.content.strip()


def plan_execute_agent(task: str) -> dict:
    """Full plan-execute cycle."""
    print(f"[TASK] {task}\n")

    # Phase 1: Plan
    print("[PHASE 1: PLANNING]")
    steps = plan(task)
    for i, step in enumerate(steps, 1):
        print(f"  Step {i}: {step}")

    # Phase 2: Execute
    print(f"\n[PHASE 2: EXECUTION]")
    context = ""
    results = []
    for i, step in enumerate(steps, 1):
        print(f"\n  Executing step {i}...")
        result = execute_step(step, context)
        context += f" {result}"
        results.append(result)
        print(f"  Done: {result[:80]}...")

    return {"task": task, "steps": steps, "results": results, "final_context": context}


if __name__ == "__main__":
    print("[PLAN-EXECUTE AGENT PATTERN]\n")
    print("=" * 60)

    task = "Create a market entry strategy for a budget ANC headphone product in India"
    result = plan_execute_agent(task)

    print(f"\n[FINAL SYNTHESIS]")
    synthesis = llm.invoke([HumanMessage(content=f"Summarize these research findings in 3 bullet points using - dashes:\n{result['final_context']}")])
    print(synthesis.content)