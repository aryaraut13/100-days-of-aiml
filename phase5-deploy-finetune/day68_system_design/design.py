import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
import random
from functools import lru_cache

print("[AI SYSTEM DESIGN — SIMULATION]\n")

# Simulate the system components
class QueryCache:
    def __init__(self):
        self._cache = {}
        self.hits   = 0
        self.misses = 0

    def get(self, key: str):
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: str):
        self._cache[key] = value

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


class QueryClassifier:
    def classify(self, query: str) -> str:
        keywords = {
            "returns":  ["return", "refund", "send back", "damaged"],
            "shipping": ["ship", "deliver", "track", "arrive", "package"],
            "payment":  ["pay", "payment", "checkout", "transaction", "upi"],
            "warranty": ["warranty", "broken", "stopped working", "repair"],
        }
        query_lower = query.lower()
        for category, words in keywords.items():
            if any(w in query_lower for w in words):
                return category
        return "general"


class MockLLM:
    def __init__(self, avg_latency: float = 1.5):
        self.avg_latency  = avg_latency
        self.calls        = 0
        self.total_tokens = 0

    def call(self, query: str) -> tuple[str, float]:
        latency = self.avg_latency + random.uniform(-0.3, 0.3)
        time.sleep(min(latency, 0.1))  # Simulate (capped for demo)
        self.calls        += 1
        self.total_tokens += 500
        return f"Response to: {query[:40]}...", latency


# Simulate 20 queries
cache      = QueryCache()
classifier = QueryClassifier()
llm        = MockLLM(avg_latency=1.5)

queries = [
    "How do I return a product?",
    "Where is my shipment?",
    "Payment failed at checkout",
    "How do I return a product?",   # duplicate -> cache hit
    "Is my product under warranty?",
    "Where is my shipment?",        # duplicate -> cache hit
    "Can I pay with UPI?",
    "How do I return a product?",   # duplicate -> cache hit
    "My package is delayed",
    "Product stopped working",
]

print(f"{'Query':45s} {'Category':10s} {'Source':8s} {'Latency':8s}")
print("-" * 75)

latencies = []

for query in queries:
    category = classifier.classify(query)
    cached   = cache.get(query)

    if cached:
        source  = "CACHE"
        latency = 0.05
        result  = cached
    else:
        source  = "LLM"
        result, latency = llm.call(query)
        cache.set(query, result)

    latencies.append(latency)
    print(f"{query[:44]:45s} {category:10s} {source:8s} {latency:.2f}s")

print("-" * 75)
print(f"\n[SYSTEM METRICS]")
print(f"Total queries:    {len(queries)}")
print(f"Cache hit rate:   {cache.hit_rate:.0%} ({cache.hits} hits / {cache.misses} misses)")
print(f"LLM calls made:   {llm.calls}")
print(f"Avg latency:      {sum(latencies)/len(latencies):.2f}s")
print(f"P95 latency:      {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
print(f"Cost saved:       {cache.hit_rate:.0%} fewer LLM calls")