# day44_multi_tool_agent/tools.py
import os
import json
import math
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()


@tool
def search_market_data(product_category: str) -> str:
    """
    Search for market data about a product category.
    Returns average price, top brands, and market size.
    """
    data = {
        "laptop": {
            "avg_price": 45000,
            "top_brands": ["Dell", "HP", "Lenovo", "Apple", "Asus"],
            "market_size": "Rs.12,000 crore",
            "growth_rate": "12% YoY",
            "top_segment": "business laptops (40% market share)"
        },
        "headphones": {
            "avg_price": 3500,
            "top_brands": ["Sony", "Bose", "JBL", "Sennheiser", "boAt"],
            "market_size": "Rs.2,500 crore",
            "growth_rate": "18% YoY",
            "top_segment": "wireless earbuds (55% market share)"
        },
        "phone": {
            "avg_price": 18000,
            "top_brands": ["Samsung", "Apple", "OnePlus", "Xiaomi", "Vivo"],
            "market_size": "Rs.1,50,000 crore",
            "growth_rate": "8% YoY",
            "top_segment": "mid-range 10-20k (45% market share)"
        },
    }
    cat = product_category.lower()
    if cat not in data:
        return f"No data for '{product_category}'. Available: {list(data.keys())}"
    d = data[cat]
    return json.dumps(d, indent=2)


@tool
def calculate_market_opportunity(
    market_size_cr: float,
    target_share_percent: float,
    avg_price: float
) -> str:
    """
    Calculate revenue opportunity given market share target.
    market_size_cr: market size in crore rupees
    target_share_percent: target market share percentage
    avg_price: average product price in Rs.
    """
    revenue_cr    = market_size_cr * (target_share_percent / 100)
    units_sold    = (revenue_cr * 1e7) / avg_price
    return (f"Target market share: {target_share_percent}% of Rs.{market_size_cr} crore market\n"
            f"Revenue opportunity: Rs.{revenue_cr:.0f} crore\n"
            f"Units to sell: {units_sold:,.0f} per year")


@tool
def write_market_report(
    category: str,
    key_findings: str,
    opportunity: str
) -> str:
    """
    Write a structured market research report from findings and opportunity data.
    """
    report = f"""
MARKET RESEARCH REPORT — {category.upper()}
{'='*50}

KEY FINDINGS:
{key_findings}

MARKET OPPORTUNITY:
{opportunity}

RECOMMENDATION:
Based on the market data, the {category} segment shows strong
growth potential. Focus on the top-performing segment for
maximum market penetration.
{'='*50}
"""
    return report.strip()