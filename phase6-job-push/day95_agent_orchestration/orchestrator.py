# day95_agent_orchestration/orchestrator.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=800
)

# Specialist agents as tools
@tool
def research_agent(query: str) -> str:
    """Research agent: finds facts and data about a topic."""
    specialist = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=300
    )
    from langchain_core.messages import HumanMessage
    response = specialist.invoke([HumanMessage(
        content=f"You are a research specialist. Find key facts about: {query}. Be specific and brief."
    )])
    return f"[RESEARCH] {response.content}"


@tool
def writing_agent(content: str) -> str:
    """Writing agent: turns research into polished content."""
    specialist = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=300
    )
    from langchain_core.messages import HumanMessage
    response = specialist.invoke([HumanMessage(
        content=f"You are a professional writer. Turn this research into a clear 2-paragraph summary:\n{content}"
    )])
    return f"[WRITTEN] {response.content}"


@tool
def critic_agent(content: str) -> str:
    """Critic agent: reviews and suggests improvements."""
    specialist = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=200
    )
    from langchain_core.messages import HumanMessage
    response = specialist.invoke([HumanMessage(
        content=f"You are an editor. Give 2 specific improvements for this content:\n{content}"
    )])
    return f"[CRITIQUE] {response.content}"


# Orchestrator agent that coordinates the specialists
orchestrator = create_react_agent(llm, [research_agent, writing_agent, critic_agent])


if __name__ == "__main__":
    print("[AGENT ORCHESTRATION — COORDINATOR + SPECIALISTS]\n")
    print("Architecture:")
    print("User -> Orchestrator -> [Research Agent | Writing Agent | Critic Agent]")
    print("=" * 60 + "\n")

    task = """Create a polished 2-paragraph brief about the Indian AI market.
    First research the topic, then write it up, then have it critiqued."""

    print(f"Task: {task}\n")
    result = orchestrator.invoke({"messages": [("human", task)]})
    print(f"\n[FINAL OUTPUT]\n{result['messages'][-1].content}")