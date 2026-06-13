# day26_llm_apis/basic_call.py
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("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"))

def chat(system_prompt: str, user_message: str,
         temperature: float = 0.7) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text


if __name__ == "__main__":
    system = "You are a concise ML assistant. Answer in 2-3 sentences only."

    questions = [
        "What is gradient descent?",
        "What is the difference between precision and recall?",
        "Why do we need cross-validation?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {chat(system, q)}")
        print("-" * 60)