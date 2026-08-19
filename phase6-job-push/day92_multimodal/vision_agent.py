# day92_multimodal/vision_agent.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import base64
import json
import requests
from dotenv import load_dotenv
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
llm    = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=500
)


def encode_image_url(url: str) -> tuple:
    """Download and encode image as base64."""
    response     = requests.get(url, timeout=10)
    image_data   = base64.standard_b64encode(response.content).decode("utf-8")
    content_type = response.headers.get("content-type", "image/jpeg")
    return image_data, content_type


def vision_call(image_url: str, prompt: str) -> str:
    """Send image + prompt to Claude and get response."""
    image_data, media_type = encode_image_url(image_url)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
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
                {"type": "text", "text": prompt}
            ],
        }]
    )
    return response.content[0].text


# Vision tools for the agent
@tool
def analyse_image(image_url: str) -> str:
    """
    Analyse an image and return a structured description.
    Use this when you need to understand what is in an image.
    Input: a valid image URL starting with http or https.
    """
    result = vision_call(
        image_url,
        """Analyse this image and respond with JSON only:
{
  "what_i_see": "brief description",
  "category": "product/person/place/other",
  "dominant_colors": ["color1", "color2"],
  "mood": "professional/casual/creative/technical",
  "suitable_for": ["use case 1", "use case 2"]
}"""
    )
    try:
        parsed = json.loads(result)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        return result


@tool
def check_image_quality(image_url: str) -> str:
    """
    Check if an image is suitable for use in a product listing or post.
    Returns quality assessment with specific improvement suggestions.
    Input: a valid image URL starting with http or https.
    """
    result = vision_call(
        image_url,
        """Rate this image for use in a product listing.
Reply with JSON only:
{
  "quality_score": 1 to 10,
  "lighting": "good/poor/average",
  "background": "clean/cluttered/neutral",
  "suitable_for_listing": true or false,
  "improvements": ["suggestion 1", "suggestion 2"]
}"""
    )
    try:
        parsed = json.loads(result)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        return result


@tool
def extract_text_from_image(image_url: str) -> str:
    """
    Extract any visible text from an image.
    Useful for reading labels, signs, or screenshots.
    Input: a valid image URL starting with http or https.
    """
    return vision_call(
        image_url,
        "Extract all visible text from this image. If no text is visible, say 'No text found'. List each text element on a new line."
    )


# Build the vision agent
tools        = [analyse_image, check_image_quality, extract_text_from_image]
vision_agent = create_react_agent(llm, tools)


def ask_vision_agent(query: str) -> str:
    result = vision_agent.invoke({"messages": [("human", query)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("[VISION AGENT — MULTIMODAL AI]\n")
    print("An AI agent that can see and reason about images.\n")
    print("=" * 60)

    # Public test images — replace with any image URL
    headphone_img = "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400"
    laptop_img    = "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=400"

    queries = [
        f"Analyse this image and tell me what product it shows and who would buy it: {headphone_img}",
        f"Is this image suitable for a product listing? Give me specific improvements: {headphone_img}",
        f"Compare these two images — which one looks more professional for an ecommerce listing? Image 1: {headphone_img} Image 2: {laptop_img}",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n[QUERY {i}]")
        print(f"Q: {query[:80]}...")
        answer = ask_vision_agent(query)
        print(f"A: {answer}\n")
        print("-" * 60)