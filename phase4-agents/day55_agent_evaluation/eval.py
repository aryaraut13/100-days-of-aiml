import os
import time
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from test_cases import TEST_CASES

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=500
)


@tool
def get_product_price(product: str) -> str:
    """Get the current price of a product."""
    prices = {"laptop": 45999, "phone": 15999, "headphones": 2999, "tablet": 25999}
    p = prices.get(product.lower())
    return f"Rs.{p}" if p else f"'{product}' not found."


@tool
def compare_products(product1: str, product2: str) -> str:
    """Compare two products by price and rating."""
    data = {
        "gaming laptop":  {"price": 75999, "rating": 4.5},
        "office laptop":  {"price": 45999, "rating": 4.2},
        "budget laptop":  {"price": 28999, "rating": 3.9},
    }
    p1 = data.get(product1.lower(), {"price": 0, "rating": 0})
    p2 = data.get(product2.lower(), {"price": 0, "rating": 0})
    return f"{product1}: Rs.{p1['price']} ({p1['rating']}*) vs {product2}: Rs.{p2['price']} ({p2['rating']}*)"


@tool
def get_recommendations(budget: float, category: str) -> str:
    """Get product recommendations within budget."""
    if category.lower() in ["audio", "headphones", "earbuds"]:
        if budget >= 2999:
            return "JBL Tune 510BT: Rs.2999 (4.2*) — Best pick under Rs.3000"
        return "Noise Buds VS104: Rs.999 (3.9*) — Budget option"
    return f"No recommendations for {category} under Rs.{budget}"


tools = [get_product_price, compare_products, get_recommendations]
agent = create_react_agent(llm, tools)


def evaluate_agent() -> None:
    print("[AGENT EVALUATION FRAMEWORK]\n")
    print(f"{'ID':6s} {'Category':15s} {'Keywords':6s} {'Pass':5s} {'Time':6s}")
    print("-" * 50)

    results = []
    for tc in TEST_CASES:
        start = time.time()
        result = agent.invoke({"messages": [("human", tc["input"])]})
        elapsed = time.time() - start
        response = result["messages"][-1].content.lower()

        # Check keyword presence
        keyword_hits = sum(1 for kw in tc["expected_keywords"] if kw.lower() in response)
        keyword_score = keyword_hits / len(tc["expected_keywords"])
        passed = keyword_score >= 0.5

        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "passed": passed,
            "keyword_score": keyword_score,
            "time": elapsed
        })

        status = "PASS" if passed else "FAIL"
        print(f"{tc['id']:6s} {tc['category']:15s} {keyword_score:.0%}    {status:5s} {elapsed:.1f}s")

    print("-" * 50)
    passed_count = sum(1 for r in results if r["passed"])
    avg_time = sum(r["time"] for r in results) / len(results)
    avg_score = sum(r["keyword_score"] for r in results) / len(results)

    print(f"\n[RESULTS]")
    print(f"  Passed:        {passed_count}/{len(TEST_CASES)}")
    print(f"  Avg Score:     {avg_score:.0%}")
    print(f"  Avg Latency:   {avg_time:.1f}s")
    print(f"  Pass Rate:     {passed_count/len(TEST_CASES):.0%}")


if __name__ == "__main__":
    evaluate_agent()