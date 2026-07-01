import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent
from typing import Any

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1000
)


class AgentDebugCallback(BaseCallbackHandler):
    """Custom callback to trace agent behavior."""

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "unknown")
        print(f"  [TOOL CALL] {tool_name}({input_str[:60]})")

    def on_tool_end(self, output: str, **kwargs) -> None:
        print(f"  [TOOL RESULT] {str(output)[:80]}...")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        print(f"  [TOOL ERROR] {str(error)}")

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        print(f"  [LLM CALL] Sending {len(prompts)} prompt(s)")

    def on_llm_end(self, response: Any, **kwargs) -> None:
        print(f"  [LLM DONE]")


@tool
def divide_numbers(a: float, b: float) -> str:
    """Divide number a by number b."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return str(a / b)


@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price for a ticker symbol.
    Only supports: RELIANCE, TCS, INFOSYS, WIPRO
    """
    stocks = {
        "RELIANCE": 2847.50,
        "TCS":      3921.00,
        "INFOSYS":  1456.75,
        "WIPRO":    456.20,
    }
    ticker = ticker.upper()
    if ticker not in stocks:
        return f"Ticker '{ticker}' not supported. Try: RELIANCE, TCS, INFOSYS, WIPRO"
    return f"{ticker}: Rs.{stocks[ticker]}"


tools = [divide_numbers, get_stock_price]
agent = create_react_agent(llm, tools)


def run_with_debug(task: str) -> None:
    print(f"\n[TASK] {task}")
    print("[TRACE]")
    result = agent.invoke({"messages": [("human", task)]})
    print(f"[ANSWER] {result['messages'][-1].content}\n")


if __name__ == "__main__":
    print("[AGENT DEBUGGING — FAILURE MODES AND HANDLING]\n")

    tasks = [
        "What is 144 divided by 12?",
        "What is the stock price of TCS and INFOSYS? Which is higher?",
        "What is the stock price of GOOGLE?",  # unsupported — agent should handle gracefully
        "What is 100 divided by 0?",            # error case
    ]

    for task in tasks:
        run_with_debug(task)
        print("-" * 55)