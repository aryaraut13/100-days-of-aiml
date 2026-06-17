
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def classify_with_few_shot(review: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        temperature=0,
        system="You classify ecommerce product reviews. Reply with one word only: POSITIVE, NEGATIVE, or NEUTRAL.",
        messages=[
            {"role": "user", "content": "This blender is incredible, smoothies in seconds!"},
            {"role": "assistant", "content": "POSITIVE"},

            {"role": "user", "content": "Stopped working after 3 uses. Returning it."},
            {"role": "assistant", "content": "NEGATIVE"},

            {"role": "user", "content": "Average product. Nothing special but does the job."},
            {"role": "assistant", "content": "NEUTRAL"},

            {"role": "user", "content": review},
        ]
    )

    return response.content[0].text.strip()


if __name__ == "__main__":

    test_reviews = [
        "Best purchase I've made this year. Highly recommend!",
        "Came damaged. Customer service was useless.",
        "It works fine I guess. Nothing to write home about.",
        "Absolutely love it, using it every single day.",
    ]

    print("[FEW-SHOT CLASSIFICATION]\n")

    for review in test_reviews:
        label = classify_with_few_shot(review)

        print(f"Review: {review}")
        print(f"Label: {label}")
        print("-" * 60)