# day92_multimodal/image_analysis.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import base64
import json
import requests
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def encode_image_from_url(url: str) -> tuple:
    """Download image and encode as base64."""
    response = requests.get(url, timeout=10)
    image_data = base64.standard_b64encode(response.content).decode("utf-8")
    content_type = response.headers.get("content-type", "image/jpeg")
    return image_data, content_type


def analyse_product_image(image_url: str) -> dict:
    """
    Analyse a product image and extract structured information.
    Returns: category, brand_clues, condition, key_features, suggested_price_range
    """
    image_data, media_type = encode_image_from_url(image_url)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type":  "image",
                    "source": {
                        "type":       "base64",
                        "media_type": media_type,
                        "data":       image_data,
                    },
                },
                {
                    "type": "text",
                    "text": """Analyse this product image and respond ONLY with valid JSON:
{
  "category": "product category",
  "condition": "new/used/refurbished",
  "key_features": ["feature1", "feature2", "feature3"],
  "color": "primary color",
  "suggested_price_range": "Rs.XXXX - Rs.XXXX",
  "confidence": 0.0 to 1.0
}"""
                }
            ],
        }]
    )

    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw, "error": "Could not parse JSON"}


def describe_image(image_url: str, question: str) -> str:
    """Ask any question about an image."""
    image_data, media_type = encode_image_from_url(image_url)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": media_type,
                        "data":       image_data,
                    },
                },
                {"type": "text", "text": question}
            ],
        }]
    )
    return response.content[0].text


if __name__ == "__main__":
    print("[MULTIMODAL AI — IMAGE ANALYSIS]\n")

    # Use a public product image for demo
    # Replace with any product image URL
    test_url = "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400"

    print("[1. STRUCTURED PRODUCT ANALYSIS]")
    print(f"Analysing image from: {test_url[:50]}...")
    result = analyse_product_image(test_url)
    print(json.dumps(result, indent=2))

    print("\n[2. CUSTOM QUESTION]")
    answer = describe_image(
        test_url,
        "What type of product is this and who would buy it? Answer in 2 sentences."
    )
    print(f"Answer: {answer}")

    print("\n[KEY INSIGHT]")
    print("Same image. Two different outputs.")
    print("Structured JSON for programmatic use.")
    print("Natural language for human consumption.")
    print("Multimodal AI handles both.")