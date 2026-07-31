# day75_project3_start/tools.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.tools import tool
import json


@tool
def search_product(product_name: str) -> str:
    """
    Search for a product and return pricing, specs, and availability.
    """
    catalog = {
        "sony wh-1000xm5": {
            "name": "Sony WH-1000XM5",
            "price_range": "Rs.20000-Rs.26000",
            "best_price": "Rs.23490 (Amazon)",
            "specs": "30hr battery, ANC, Bluetooth 5.2, 254g",
            "availability": "In stock on Amazon, Flipkart",
            "variants": ["Black", "Silver", "Midnight Blue"],
        },
        "boat rockerz 450": {
            "name": "boAt Rockerz 450",
            "price_range": "Rs.999-Rs.1699",
            "best_price": "Rs.999 (Amazon)",
            "specs": "40hr battery, Bluetooth 5.0, 220g, with mic",
            "availability": "In stock everywhere",
            "variants": ["Black", "Blue", "Red", "White"],
        },
        "apple airpods pro": {
            "name": "Apple AirPods Pro (2nd gen)",
            "price_range": "Rs.22000-Rs.26000",
            "best_price": "Rs.22900 (Flipkart)",
            "specs": "30hr total, ANC, USB-C, H2 chip",
            "availability": "In stock at Apple Store, Amazon",
            "variants": ["White only"],
        },
        "jbl tune 510bt": {
            "name": "JBL Tune 510BT",
            "price_range": "Rs.2000-Rs.3500",
            "best_price": "Rs.2199 (Amazon)",
            "specs": "40hr battery, Bluetooth 5.0, JBL Pure Bass",
            "availability": "In stock",
            "variants": ["Black", "Blue", "White", "Pink"],
        },
    }

    key    = product_name.lower().strip()
    result = catalog.get(key)

    if not result:
        for k in catalog:
            if any(word in k for word in key.split()):
                result = catalog[k]
                break

    if not result:
        return f"Product '{product_name}' not found. Try: Sony WH-1000XM5, boAt Rockerz 450, Apple AirPods Pro, JBL Tune 510BT"

    return json.dumps(result, indent=2)


@tool
def get_product_reviews(product_name: str) -> str:
    """
    Get customer reviews summary for a product.
    Returns overall sentiment, top praises, and common complaints.
    """
    reviews = {
        "sony wh-1000xm5": {
            "rating":     4.6,
            "total":      "12,847 ratings",
            "sentiment":  "Very Positive",
            "praises":    ["Best ANC available", "Excellent sound quality", "30hr battery", "Premium build", "Great call quality"],
            "complaints": ["Expensive", "No IP rating", "Non-foldable design", "Limited India-specific features"],
            "verdict":    "Best-in-class ANC headphone. Worth every rupee if audio quality is priority.",
        },
        "boat rockerz 450": {
            "rating":     4.1,
            "total":      "45,230 ratings",
            "sentiment":  "Positive",
            "praises":    ["Great value", "Good bass", "40hr battery", "Comfortable for long use"],
            "complaints": ["Build quality after 6 months", "Mic quality average", "No ANC", "Cheap plastic feel"],
            "verdict":    "Best budget option. Great for the price but don't expect premium feel.",
        },
        "apple airpods pro": {
            "rating":     4.7,
            "total":      "24,891 ratings",
            "sentiment":  "Very Positive",
            "praises":    ["Seamless iOS integration", "Excellent ANC", "Comfortable fit", "Transparency mode"],
            "complaints": ["Only for Apple users", "Expensive", "White color only", "Battery degrades over time"],
            "verdict":    "Perfect for iPhone users. Ecosystem integration is unmatched.",
        },
        "jbl tune 510bt": {
            "name": "JBL Tune 510BT",
            "rating":     4.2,
            "total":      "28,102 ratings",
            "sentiment":  "Positive",
            "praises":    ["JBL sound signature", "Lightweight", "Good battery", "Wide color options"],
            "complaints": ["No ANC", "Average mic", "Plastic build", "Basic features only"],
            "verdict":    "Solid mid-range choice. Good sound, basic features.",
        },
    }

    key    = product_name.lower().strip()
    result = reviews.get(key)

    if not result:
        for k in reviews:
            if any(word in k for word in key.split()):
                result = reviews[k]
                break

    if not result:
        return f"No reviews found for '{product_name}'"

    return json.dumps(result, indent=2)


@tool
def compare_products(product1: str, product2: str) -> str:
    """
    Compare two products side by side.
    Returns a structured comparison with a recommendation.
    """
    prices = {
        "sony wh-1000xm5":  {"price": 23490, "rating": 4.6, "anc": True,  "battery": 30, "category": "premium"},
        "boat rockerz 450": {"price": 999,   "rating": 4.1, "anc": False, "battery": 40, "category": "budget"},
        "apple airpods pro": {"price": 22900, "rating": 4.7, "anc": True,  "battery": 30, "category": "premium"},
        "jbl tune 510bt":   {"price": 2199,  "rating": 4.2, "anc": False, "battery": 40, "category": "mid-range"},
    }

    p1 = prices.get(product1.lower())
    p2 = prices.get(product2.lower())

    if not p1 or not p2:
        return f"Could not find one or both products for comparison."

    price_diff   = abs(p1["price"] - p2["price"])
    cheaper      = product1 if p1["price"] < p2["price"] else product2
    higher_rated = product1 if p1["rating"] > p2["rating"] else product2

    return json.dumps({
        product1: p1,
        product2: p2,
        "price_difference": f"Rs.{price_diff}",
        "cheaper":          cheaper,
        "higher_rated":     higher_rated,
        "recommendation":   f"If budget is priority: {cheaper}. If quality is priority: {higher_rated}.",
    }, indent=2)


@tool
def generate_buying_guide(budget: float, use_case: str) -> str:
    """
    Generate a personalized buying guide based on budget and use case.
    use_case options: music, calls, gaming, casual, exercise
    """
    guides = {
        "budget_music": {
            "top_pick":    "boAt Rockerz 450 (Rs.999)",
            "runner_up":   "JBL Tune 510BT (Rs.2199)",
            "reasoning":   "Best bass and battery at budget price points",
            "avoid":       "Don't expect ANC at this budget",
        },
        "premium_music": {
            "top_pick":    "Sony WH-1000XM5 (Rs.23490)",
            "runner_up":   "Apple AirPods Pro (Rs.22900) for iPhone users",
            "reasoning":   "Industry-leading ANC and sound quality",
            "avoid":       "Don't buy premium if you mostly use during commute without ANC",
        },
        "calls": {
            "top_pick":    "Sony WH-1000XM5 (Rs.23490) — best call quality",
            "budget_pick": "boAt Rockerz 450 (Rs.999) — decent mic for the price",
            "reasoning":   "Beamforming mics in Sony reduce background noise significantly",
            "avoid":       "JBL Tune 510BT mic is mediocre for professional calls",
        },
    }

    if budget < 2000:
        key = "budget_music"
    elif budget < 10000:
        key = "budget_music" if use_case in ["casual", "exercise"] else "calls"
    else:
        key = "premium_music" if use_case in ["music", "casual"] else "calls"

    guide = guides.get(key, guides["budget_music"])
    guide["budget"]   = f"Rs.{budget:.0f}"
    guide["use_case"] = use_case
    return json.dumps(guide, indent=2)