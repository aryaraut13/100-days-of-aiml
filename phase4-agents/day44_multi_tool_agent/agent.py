import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from tools import search_market_data, calculate_market_opportunity, write_market_report

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=2000
)

tools = [search_market_data, calculate_market_opportunity, write_market_report]
agent = create_react_agent(llm, tools)


def research(task: str) -> str:
    result = agent.invoke({"messages": [("human", task)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("[MULTI-TOOL MARKET RESEARCH AGENT]\n")

    task = """Research the headphone market and create a report.
    Include: market size, top brands, growth rate, and calculate
    the revenue opportunity if we capture 2% market share
    at an average price of Rs.3500."""

    print(f"Task: {task}\n")
    print("Agent working...\n")
    report = research(task)
    print(report)