# day74_llm_optimization/streaming.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def non_streaming(prompt: str) -> tuple:
    """Standard call — waits for complete response."""
    start    = time.time()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = (time.time() - start) * 1000
    return response.content[0].text, latency


def streaming(prompt: str) -> tuple:
    """Streaming call — tokens arrive as generated."""
    start        = time.time()
    first_token  = None
    full_text    = ""
    token_count  = 0

    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            if first_token is None:
                first_token = (time.time() - start) * 1000
            full_text   += text
            token_count += 1

    total_latency = (time.time() - start) * 1000
    return full_text, first_token, total_latency, token_count


if __name__ == "__main__":
    print("[STREAMING vs NON-STREAMING COMPARISON]\n")

    prompt = "Explain what a RAG pipeline is in 3 sentences."

    print("[NON-STREAMING]")
    print("Waiting for complete response...")
    text, latency = non_streaming(prompt)
    print(f"Time to first word: {latency:.0f}ms (full response)")
    print(f"Response: {text[:100]}...\n")

    print("[STREAMING]")
    print("Tokens arriving as generated: ", end="", flush=True)
    text, first_token, total, tokens = streaming(prompt)
    print(f"\nTime to first token: {first_token:.0f}ms")
    print(f"Total time:          {total:.0f}ms")
    print(f"Tokens streamed:     {tokens}")
    print(f"\n[KEY INSIGHT]")
    print(f"Non-streaming: user waits {latency:.0f}ms before seeing anything")
    print(f"Streaming: user sees first token in {first_token:.0f}ms")
    print(f"Perceived latency reduction: ~{(latency-first_token)/latency*100:.0f}%")