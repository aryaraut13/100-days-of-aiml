import json
from langchain_core.tools import tool


@tool
def search_ecommerce_products(query: str, max_results: int = 5) -> str:
    """
    Search for ecommerce products matching a query.
    Returns product name, brand, price, rating, and reviews count.
    """
    # Simulated ecommerce database
    database = [
        {"name": "boAt Rockerz 450",       "brand": "boAt",       "category": "headphones", "price": 1299,  "rating": 4.1, "reviews": 45000},
        {"name": "Sony WH-1000XM5",        "brand": "Sony",       "category": "headphones", "price": 24990, "rating": 4.6, "reviews": 12000},
        {"name": "JBL Tune 510BT",         "brand": "JBL",        "category": "headphones", "price": 2999,  "rating": 4.2, "reviews": 28000},
        {"name": "Sennheiser HD 450BT",    "brand": "Sennheiser", "category": "headphones", "price": 7990,  "rating": 4.4, "reviews": 8500},
        {"name": "Noise Buds VS104",       "brand": "Noise",      "category": "headphones", "price": 999,   "rating": 3.9, "reviews": 32000},
        {"name": "OnePlus Nord Buds 2",    "brand": "OnePlus",    "category": "earbuds",    "price": 2799,  "rating": 4.3, "reviews": 15000},
        {"name": "Samsung Galaxy Buds2",   "brand": "Samsung",    "category": "earbuds",    "price": 5999,  "rating": 4.2, "reviews": 9000},
        {"name": "Apple AirPods Pro",      "brand": "Apple",      "category": "earbuds",    "price": 24900, "rating": 4.7, "reviews": 25000},
    ]

    query_lower = query.lower()
    results = [
        p for p in database
        if query_lower in p["name"].lower()
        or query_lower in p["brand"].lower()
        or query_lower in p["category"].lower()
    ][:max_results]

    if not results:
        return f"No products found for '{query}'"

    output = []
    for p in results:
        output.append(
            f"{p['name']} | Brand: {p['brand']} | "
            f"Price: Rs.{p['price']} | Rating: {p['rating']} | "
            f"Reviews: {p['reviews']:,}"
        )
    return "\n".join(output)


@tool
def analyze_price_segments(category: str) -> str:
    """
    Analyze price segments for a product category.
    Returns budget/mid/premium breakdown with market share estimates.
    """
    segments = {
        "headphones": {
            "budget":  {"range": "Under Rs.2000",          "share": "45%", "leaders": ["boAt", "Noise", "Skullcandy"]},
            "mid":     {"range": "Rs.2000 - Rs.10000",     "share": "35%", "leaders": ["JBL", "Sony", "Sennheiser"]},
            "premium": {"range": "Above Rs.10000",         "share": "20%", "leaders": ["Sony", "Bose", "Apple"]},
        },
        "earbuds": {
            "budget":  {"range": "Under Rs.1500",          "share": "40%", "leaders": ["boAt", "Noise", "PTron"]},
            "mid":     {"range": "Rs.1500 - Rs.5000",      "share": "38%", "leaders": ["OnePlus", "Samsung", "Realme"]},
            "premium": {"range": "Above Rs.5000",          "share": "22%", "leaders": ["Apple", "Samsung", "Sony"]},
        },
        "laptop": {
            "budget":  {"range": "Under Rs.35000",         "share": "30%", "leaders": ["Lenovo", "Acer", "HP"]},
            "mid":     {"range": "Rs.35000 - Rs.70000",    "share": "45%", "leaders": ["Dell", "HP", "Lenovo"]},
            "premium": {"range": "Above Rs.70000",         "share": "25%", "leaders": ["Apple", "Dell", "Asus"]},
        },
    }

    cat = category.lower()
    if cat not in segments:
        return f"Category '{category}' not found. Available: {list(segments.keys())}"

    result = [f"PRICE SEGMENTS — {category.upper()}"]
    for seg, data in segments[cat].items():
        result.append(
            f"\n{seg.upper()} ({data['range']}):\n"
            f"  Market share: {data['share']}\n"
            f"  Key players: {', '.join(data['leaders'])}"
        )
    return "\n".join(result)


@tool
def get_market_trends(category: str) -> str:
    """
    Get current market trends and growth data for a product category.
    """
    trends = {
        "headphones": {
            "market_size":    "Rs.3,200 crore (India, 2024)",
            "growth_rate":    "22% CAGR",
            "key_trends":     ["ANC adoption up 35% YoY", "Wireless now 70% of sales", "Gaming headsets fastest growing"],
            "opportunities":  ["Sub-Rs.2000 ANC segment underserved", "Sports/fitness segment growing", "Kids headphones niche"],
        },
        "earbuds": {
            "market_size":    "Rs.4,800 crore (India, 2024)",
            "growth_rate":    "35% CAGR",
            "key_trends":     ["TWS dominates 80% share", "ANC in sub-Rs.3000 growing", "Health features (HearID, EQ)"],
            "opportunities":  ["Enterprise/business segment", "Hearing aid adjacent products", "Gaming earbuds"],
        },
        "laptop": {
            "market_size":    "Rs.18,000 crore (India, 2024)",
            "growth_rate":    "12% CAGR",
            "key_trends":     ["AI PC category emerging", "Thin & light dominates", "Work-from-home demand steady"],
            "opportunities":  ["AI-powered laptops", "Creator segment", "Education sector"],
        },
    }

    cat = category.lower()
    if cat not in trends:
        return f"No trend data for '{category}'. Available: {list(trends.keys())}"

    t = trends[cat]
    result = [
        f"MARKET TRENDS — {category.upper()}",
        f"Market Size: {t['market_size']}",
        f"Growth Rate: {t['growth_rate']}",
        f"\nKey Trends:",
    ]
    for trend in t["key_trends"]:
        result.append(f"  • {trend}")
    result.append("\nOpportunities:")
    for opp in t["opportunities"]:
        result.append(f"  • {opp}")

    return "\n".join(result)