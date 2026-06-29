import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1000
)


@tool
def search_products(query: str) -> str:
    """
    Search for products matching a query.
    Returns a list of matching products with prices.
    """
    catalog = [
        {"name": "Gaming Laptop",    "price": 75999, "category": "laptop",     "rating": 4.5},
        {"name": "Office Laptop",    "price": 45999, "category": "laptop",     "rating": 4.2},
        {"name": "Budget Laptop",    "price": 28999, "category": "laptop",     "rating": 3.9},
        {"name": "Noise-cancelling Headphones", "price": 8999, "category": "audio", "rating": 4.6},
        {"name": "Budget Headphones", "price": 999,  "category": "audio",     "rating": 3.8},
        {"name": "Wireless Mouse",   "price": 1299,  "category": "accessory", "rating": 4.3},
        {"name": "Mechanical Keyboard", "price": 3499, "category": "accessory", "rating": 4.7},
    ]
    query_lower = query.lower()
    results = [p for p in catalog if query_lower in p["name"].lower()
               or query_lower in p["category"].lower()]
    if not results:
        return "No products found matching your query."
    output = []
    for p in results:
        output.append(f"{p['name']}: Rs.{p['price']} | Rating: {p['rating']}/5")
    return "\n".join(output)


@tool
def compare_products(product1: str, product2: str) -> str:
    """
    Compare two products by name and return a comparison summary.
    """
    prices = {
        "gaming laptop": 75999, "office laptop": 45999, "budget laptop": 28999,
        "noise-cancelling headphones": 8999, "budget headphones": 999,
        "wireless mouse": 1299, "mechanical keyboard": 3499,
    }
    ratings = {
        "gaming laptop": 4.5, "office laptop": 4.2, "budget laptop": 3.9,
        "noise-cancelling headphones": 4.6, "budget headphones": 3.8,
        "wireless mouse": 4.3, "mechanical keyboard": 4.7,
    }
    p1, p2 = product1.lower(), product2.lower()
    if p1 not in prices or p2 not in prices:
        return "One or both products not found. Please use exact product names."

    price_diff = abs(prices[p1] - prices[p2])
    better_value = p1 if ratings[p1]/prices[p1] > ratings[p2]/prices[p2] else p2

    return (f"{product1}: Rs.{prices[p1]} | Rating: {ratings[p1]}\n"
            f"{product2}: Rs.{prices[p2]} | Rating: {ratings[p2]}\n"
            f"Price difference: Rs.{price_diff}\n"
            f"Better value: {better_value.title()}")


@tool
def get_recommendations(budget: float, category: str) -> str:
    """
    Get product recommendations within a budget for a given category.
    Categories: laptop, audio, accessory
    """
    catalog = {
        "laptop": [
            ("Gaming Laptop", 75999, 4.5),
            ("Office Laptop", 45999, 4.2),
            ("Budget Laptop", 28999, 3.9),
        ],
        "audio": [
            ("Noise-cancelling Headphones", 8999, 4.6),
            ("Budget Headphones", 999, 3.8),
        ],
        "accessory": [
            ("Mechanical Keyboard", 3499, 4.7),
            ("Wireless Mouse", 1299, 4.3),
        ],
    }
    cat = category.lower()
    if cat not in catalog:
        return f"Category '{category}' not found. Try: laptop, audio, accessory"

    affordable = [(n, p, r) for n, p, r in catalog[cat] if p <= budget]
    if not affordable:
        return f"No {category} products found under Rs.{budget}"

    best = max(affordable, key=lambda x: x[2])
    results = "\n".join([f"  {n}: Rs.{p} | Rating: {r}" for n, p, r in affordable])
    return f"Products under Rs.{budget}:\n{results}\nBest pick: {best[0]} (Rating: {best[2]})"


tools  = [search_products, compare_products, get_recommendations]
agent  = create_react_agent(llm, tools)

system_prompt = """You are a helpful ecommerce shopping assistant.
Help users find products, compare options, and make purchase decisions.
Always use the available tools to get accurate information."""


def ask_agent(question: str) -> str:
    result = agent.invoke({
        "messages": [("system", system_prompt), ("human", question)]
    })
    return result["messages"][-1].content


if __name__ == "__main__":
    print("[ECOMMERCE SHOPPING AGENT]\n")

    questions = [
        "I need a laptop under Rs.50000. What do you recommend?",
        "Compare gaming laptop vs office laptop",
        "What audio products do you have under Rs.5000?",
    ]

    for q in questions:
        print(f"User: {q}")
        answer = ask_agent(q)
        print(f"Agent: {answer}\n")
        print("-" * 55)