import os
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


@tool
def search_reviews(product_name: str) -> str:
    """
    Search for customer reviews and sentiment for a product.
    Returns common praises, complaints, and overall sentiment.
    """
    reviews_db = {
        "boat rockerz 450": {
            "sentiment": "Mostly positive",
            "avg_rating": 4.1,
            "total_reviews": 45000,
            "praises": ["Good bass", "Comfortable for long hours", "Value for money", "40hr battery"],
            "complaints": ["Mic quality", "Build quality over time", "ANC not available"],
            "summary": "Best-selling budget headphone. Great value but build quality concerns after 6 months."
        },
        "sony wh-1000xm5": {
            "sentiment": "Very positive",
            "avg_rating": 4.6,
            "total_reviews": 12000,
            "praises": ["Best ANC available", "30hr battery", "Premium build", "Excellent call quality"],
            "complaints": ["Expensive", "No IP rating", "Foldable design removed"],
            "summary": "Industry benchmark for ANC headphones. Premium price but worth every rupee."
        },
        "jbl tune 510bt": {
            "sentiment": "Positive",
            "avg_rating": 4.2,
            "total_reviews": 28000,
            "praises": ["Clear sound", "Lightweight", "Good battery", "JBL app"],
            "complaints": ["No ANC", "Average mic", "Plastic build"],
            "summary": "Solid mid-range option. Good sound but feels budget at this price."
        },
    }

    key = product_name.lower()
    if key not in reviews_db:
        return f"No review data for '{product_name}'. Try: boAt Rockerz 450, Sony WH-1000XM5, JBL Tune 510BT"

    r = reviews_db[key]
    return (
        f"REVIEWS — {product_name}\n"
        f"Sentiment: {r['sentiment']} | Avg Rating: {r['avg_rating']} | Total: {r['total_reviews']:,}\n"
        f"Praises: {', '.join(r['praises'])}\n"
        f"Complaints: {', '.join(r['complaints'])}\n"
        f"Summary: {r['summary']}"
    )


if __name__ == "__main__":
    from langchain_anthropic import ChatAnthropic
    from langgraph.prebuilt import create_react_agent
    from report_tool import analyze_competitor, write_research_report

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2000
    )

    tools = [search_reviews, analyze_competitor, write_research_report]
    agent = create_react_agent(llm, tools)

    task = """Create a competitive analysis report for headphones.
    1. Search reviews for boAt Rockerz 450 and Sony WH-1000XM5
    2. Analyze both as competitors
    3. Write a research report with your findings"""

    print("[PROJECT 2 — COMPETITIVE ANALYSIS]\n")
    result = agent.invoke({"messages": [("human", task)]})
    print(result["messages"][-1].content)