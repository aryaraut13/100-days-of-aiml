import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def standard_prompt(question: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text


def chain_of_thought(question: str) -> str:
    cot = f"Think through this step by step before giving your final answer.\n\nQuestion: {question}"
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        temperature=0,
        messages=[{"role": "user", "content": cot}]
    )
    return response.content[0].text


if __name__ == "__main__":
    question = """A store sells apples for Rs.15 each and oranges for Rs.20 each.
    A customer buys 3 apples and 4 oranges and pays with Rs.200.
    How much change do they receive?"""

    print("[STANDARD PROMPT]")
    print(standard_prompt(question))

    print("\n" + "="*60)
    print("[CHAIN OF THOUGHT]")
    print(chain_of_thought(question))

    print("\n" + "="*60)
    print("CoT forces step-by-step reasoning.")
    print("On complex problems this reduces errors significantly.")