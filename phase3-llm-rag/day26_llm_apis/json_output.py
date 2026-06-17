import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def get_structured_output(text: str) -> dict:
    system = """
You are a text analysis API.

Return ONLY valid JSON.
No explanations.
No markdown.
No code fences.

Format:
{
  "sentiment": "positive",
  "topic": "string",
  "confidence": 0.95
}
"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        system=system,
        messages=[
            {
                "role": "user",
                "content": f"Analyse this review and return JSON only: {text}"
            }
        ]
    )

    raw = response.content[0].text.strip()

    # Remove markdown code fences if Claude adds them
    raw = raw.replace("```json", "")
    raw = raw.replace("```", "")
    raw = raw.strip()

    return json.loads(raw)


if __name__ == "__main__":

    reviews = [
        "This product is absolutely amazing, works perfectly!",
        "Terrible quality, broke after one day. Waste of money.",
        "It's okay. Does what it says, nothing special."
    ]

    print("[STRUCTURED OUTPUT FROM LLM]\n")

    for review in reviews:
        result = get_structured_output(review)

        print(f"Review: {review}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Topic: {result['topic']}")
        print(f"Confidence: {result['confidence']}")
        print("-" * 60)
        