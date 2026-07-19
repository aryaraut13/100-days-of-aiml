import os
import json
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=500
)

# Bind tools to the LLM
@tool
def get_product_info(product_name: str) -> str:
    """
    Get information about an ecommerce product.
    Returns price, rating, and availability.
    """
    # Simulated product database
    products = {
        "laptop":     {"price": 45999, "rating": 4.3, "stock": "in stock"},
        "headphones": {"price": 2999,  "rating": 4.5, "stock": "in stock"},
        "phone":      {"price": 15999, "rating": 4.1, "stock": "out of stock"},
        "tablet":     {"price": 25999, "rating": 4.4, "stock": "in stock"},
    }
    key = product_name.lower()
    if key in products:
        p = products[key]
        return f"{product_name}: Rs.{p['price']} | Rating: {p['rating']}/5 | {p['stock']}"
    return f"Product '{product_name}' not found in database."


@tool
def calculate_discount(price: float, discount_percent: float) -> str:
    """
    Calculate the discounted price and savings amount.
    """
    discount = price * (discount_percent / 100)
    final    = price - discount
    return f"Original: Rs.{price} | Discount: Rs.{discount:.0f} ({discount_percent}%) | Final: Rs.{final:.0f}"


tools = [get_product_info, calculate_discount]
llm_with_tools = llm.bind_tools(tools)


if __name__ == "__main__":
    print("[CUSTOM TOOLS — LLM WITH TOOL BINDING]\n")

    # Show tool definitions
    for t in tools:
        print(f"Tool: {t.name}")
        print(f"  {t.description}\n")

    # Direct tool calls
    print("[DIRECT TOOL CALLS]\n")
    print(get_product_info.invoke("laptop"))
    print(get_product_info.invoke("phone"))
    print(calculate_discount.invoke({"price": 45999, "discount_percent": 15}))

    # LLM decides which tool to call
    print("\n[LLM TOOL SELECTION]\n")
    response = llm_with_tools.invoke([
        HumanMessage("What is the price of headphones and what would it cost with 20% discount?")
    ])
    print(f"Tool calls requested by LLM: {len(response.tool_calls)}")
    for tc in response.tool_calls:
        print(f"  -> {tc['name']}({tc['args']})")