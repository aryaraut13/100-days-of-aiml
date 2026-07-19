import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1000
)


@tool
def get_product_price(product: str) -> str:
    """Get the current price of a product."""
    prices = {
        "laptop": 45999, "phone": 15999,
        "headphones": 2999, "tablet": 25999,
    }
    p = prices.get(product.lower())
    return f"Rs.{p}" if p else f"'{product}' not found."


@tool
def add_to_wishlist(product: str, reason: str) -> str:
    """Add a product to the user's wishlist with a reason."""
    return f"Added '{product}' to wishlist. Reason: {reason}"


@tool
def view_wishlist() -> str:
    """View the current wishlist."""
    return "Wishlist is managed in conversation memory."


tools    = [get_product_price, add_to_wishlist, view_wishlist]
memory   = MemorySaver()
agent    = create_react_agent(llm, tools, checkpointer=memory)
config   = {"configurable": {"thread_id": "user_session_1"}}


def chat(message: str) -> str:
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    print("[AGENT WITH MEMORY]\n")

    conversation = [
        "My name is Arya and my budget is Rs.20000.",
        "What laptops are available? Can you check the price?",
        "Add the laptop to my wishlist — I like the specs.",
        "What's my budget again and what did I add to my wishlist?",
    ]

    for msg in conversation:
        print(f"User:  {msg}")
        reply = chat(msg)
        print(f"Agent: {reply}\n")