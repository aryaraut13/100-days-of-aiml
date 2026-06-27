import os
import math
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1000
)


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use Python syntax."""
    try:
        result = eval(expression, {"math": math, "__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_product_price(product: str) -> str:
    """Get the current price of a product in Rs."""
    prices = {
        "laptop": 45999, "phone": 15999,
        "headphones": 2999, "tablet": 25999,
        "keyboard": 1499, "mouse": 799,
    }
    p = prices.get(product.lower())
    return f"Rs.{p}" if p else f"Product '{product}' not found."


@tool
def apply_gst(amount: float, rate: float = 18.0) -> str:
    """Calculate price after adding GST. Default GST rate is 18%."""
    gst    = amount * (rate / 100)
    total  = amount + gst
    return f"Base: Rs.{amount} | GST ({rate}%): Rs.{gst:.0f} | Total: Rs.{total:.0f}"


tools = [calculator, get_product_price, apply_gst]
agent = create_react_agent(llm, tools)


def run_agent(task: str) -> None:
    print(f"[TASK] {task}\n")
    result = agent.invoke({"messages": [("human", task)]})
    # Get the final message
    final = result["messages"][-1].content
    print(f"[ANSWER] {final}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("ReAct AGENT — Reason + Act Loop")
    print("=" * 60 + "\n")

    tasks = [
        "What is the price of a laptop with 18% GST included?",
        "If I buy 3 phones and 2 headphones, what is the total cost before GST?",
        "What is 15% of the tablet price?",
    ]

    for task in tasks:
        run_agent(task)
        print("-" * 60)