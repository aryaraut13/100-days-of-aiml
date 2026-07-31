# day74_llm_optimization/caching.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import hashlib
import json
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=200
)


class LLMCache:
    """Simple in-memory LLM response cache."""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache   = {}
        self.ttl      = ttl_seconds
        self.hits     = 0
        self.misses   = 0

    def _key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    def get(self, prompt: str):
        key    = self._key(prompt)
        entry  = self._cache.get(key)
        if entry and (time.time() - entry["ts"]) < self.ttl:
            self.hits += 1
            return entry["response"]
        self.misses += 1
        return None

    def set(self, prompt: str, response: str):
        self._cache[self._key(prompt)] = {
            "response": response,
            "ts":       time.time()
        }

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


cache = LLMCache(ttl_seconds=300)


def cached_llm_call(prompt: str) -> tuple:
    """Call LLM with caching. Returns (response, latency_ms, cached)."""
    cached = cache.get(prompt)
    if cached:
        return cached, 0.5, True

    start    = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    latency  = (time.time() - start) * 1000
    result   = response.content

    cache.set(prompt, result)
    return result, latency, False


if __name__ == "__main__":
    print("[LLM CACHING — PERFORMANCE COMPARISON]\n")

    prompts = [
        "What is RAG in one sentence?",
        "What is LangChain in one sentence?",
        "What is RAG in one sentence?",       # cache hit
        "What is fine-tuning in one sentence?",
        "What is LangChain in one sentence?",  # cache hit
        "What is RAG in one sentence?",        # cache hit
    ]

    print(f"{'Prompt':45s} {'Source':8s} {'Latency':10s}")
    print("-" * 65)

    total_saved = 0
    for prompt in prompts:
        response, latency, cached = cached_llm_call(prompt)
        source = "CACHE" if cached else "LLM"
        if cached:
            total_saved += 1500  # avg LLM latency saved
        print(f"{prompt[:44]:45s} {source:8s} {latency:.0f}ms")

    print("-" * 65)
    print(f"\n[RESULTS]")
    print(f"Cache hit rate:     {cache.hit_rate:.0%} ({cache.hits}/{cache.hits+cache.misses})")
    print(f"LLM calls made:     {cache.misses}")
    print(f"Calls avoided:      {cache.hits}")
    print(f"Latency saved:      ~{total_saved}ms total")
    print(f"Cost saved:         ~{cache.hits * 0.000175 * 100:.2f} paise per session")