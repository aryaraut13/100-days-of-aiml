import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def zero_shot(task: str, input_text: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        system=task,
        messages=[{"role": "user", "content": input_text}]
    )
    return response.content[0].text


def few_shot(review: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        temperature=0,
        system="Classify the sentiment. Reply with one word: POSITIVE, NEGATIVE, or NEUTRAL.",
        messages=[
            {"role": "user",      "content": "This is incredible, best product ever!"},
            {"role": "assistant", "content": "POSITIVE"},
            {"role": "user",      "content": "Complete waste of money. Broke immediately."},
            {"role": "assistant", "content": "NEGATIVE"},
            {"role": "user",      "content": "It's fine. Does the job."},
            {"role": "assistant", "content": "NEUTRAL"},
            {"role": "user",      "content": review},
        ]
    )
    return response.content[0].text.strip()


if __name__ == "__main__":
    # Zero-shot
    print("[ZERO-SHOT]\n")
    task = "Summarise the following text in exactly one sentence."
    text = """Machine learning is a subset of artificial intelligence that gives
    systems the ability to automatically learn and improve from experience without
    being explicitly programmed. It focuses on developing computer programs that
    can access data and use it to learn for themselves."""
    print(f"Input: {text[:80]}...")
    print(f"Output: {zero_shot(task, text)}")

    # Few-shot
    print("\n[FEW-SHOT]\n")
    reviews = [
        "Best purchase I've made this year!",
        "Arrived broken. Very disappointed.",
        "Works as described. Nothing more.",
    ]
    for r in reviews:
        print(f"{few_shot(r):10s} | {r}")