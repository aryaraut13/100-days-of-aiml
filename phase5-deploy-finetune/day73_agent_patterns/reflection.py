# day73_agent_patterns/reflection.py
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
    max_tokens=800
)


def generate(task: str) -> str:
    """Initial generation."""
    response = llm.invoke([HumanMessage(content=f"Complete this task in 3-4 sentences:\n{task}")])
    return response.content.strip()


def reflect(task: str, output: str) -> str:
    """Reflect on the output — identify weaknesses."""
    response = llm.invoke([HumanMessage(content=f"""
You are a critical reviewer. Identify 2-3 specific weaknesses in this output.
Be concise. Each weakness on one line starting with -.

Task: {task}
Output: {output}

Weaknesses:""")])
    return response.content.strip()


def improve(task: str, output: str, critique: str) -> str:
    """Improve based on critique."""
    response = llm.invoke([HumanMessage(content=f"""
Improve this output based on the critique. Keep it to 3-4 sentences.

Task: {task}
Original output: {output}
Critique: {critique}

Improved output:""")])
    return response.content.strip()


def reflection_agent(task: str, iterations: int = 2) -> dict:
    """Run generate-reflect-improve loop."""
    print(f"[TASK] {task}\n")

    current = generate(task)
    print(f"[GENERATION 0 — INITIAL]\n{current}\n")

    history = [current]

    for i in range(iterations):
        print(f"[REFLECTION {i+1}]")
        critique = reflect(task, current)
        print(f"{critique}\n")

        print(f"[GENERATION {i+1} — IMPROVED]")
        current = improve(task, current, critique)
        print(f"{current}\n")
        history.append(current)

    return {"task": task, "iterations": iterations, "history": history, "final": current}


if __name__ == "__main__":
    print("[REFLECTION AGENT PATTERN]\n")
    print("=" * 60 + "\n")

    task = "Write a one-paragraph pitch for an AI-powered ecommerce market research tool"
    result = reflection_agent(task, iterations=2)

    print(f"[SUMMARY]")
    print(f"Iterations: {result['iterations']}")
    print(f"Generations: {len(result['history'])}")
    print(f"\nImprovement from generation 0 to {result['iterations']}:")
    print(f"Initial length: {len(result['history'][0])} chars")
    print(f"Final length:   {len(result['final'])} chars")