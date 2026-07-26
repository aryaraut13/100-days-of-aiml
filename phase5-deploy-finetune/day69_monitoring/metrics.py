# day69_monitoring/metrics.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
import random
from collections import defaultdict, deque
from datetime import datetime


class MetricsCollector:
    """Lightweight metrics collector for AI applications."""

    def __init__(self, window_seconds: int = 60):
        self.window   = window_seconds
        self.counters = defaultdict(int)
        self.timings  = defaultdict(list)
        self.gauges   = {}

    def increment(self, metric: str, value: int = 1):
        self.counters[metric] += value

    def timing(self, metric: str, value_ms: float):
        self.timings[metric].append(value_ms)

    def gauge(self, metric: str, value: float):
        self.gauges[metric] = value

    def percentile(self, metric: str, p: float) -> float:
        values = sorted(self.timings.get(metric, [0]))
        idx    = int(len(values) * p / 100)
        return values[min(idx, len(values) - 1)]

    def report(self):
        print("\n[METRICS REPORT]")
        print(f"Timestamp: {datetime.utcnow().isoformat()[:19]}\n")

        print("COUNTERS:")
        for k, v in self.counters.items():
            print(f"  {k:30s}: {v}")

        print("\nLATENCY (ms):")
        for k, values in self.timings.items():
            avg = sum(values) / len(values)
            p95 = self.percentile(k, 95)
            p99 = self.percentile(k, 99)
            print(f"  {k:30s}: avg={avg:.0f} | p95={p95:.0f} | p99={p99:.0f}")

        print("\nGAUGES:")
        for k, v in self.gauges.items():
            print(f"  {k:30s}: {v}")


if __name__ == "__main__":
    print("[METRICS COLLECTION DEMO]\n")

    metrics = MetricsCollector()

    # Simulate 50 requests
    categories = ["returns", "shipping", "payment", "warranty"]
    for i in range(50):
        cat     = random.choice(categories)
        latency = random.gauss(1400, 300)
        cached  = random.random() < 0.3

        metrics.increment("requests_total")
        metrics.increment(f"requests_{cat}")
        metrics.timing("llm_latency_ms", latency if not cached else 45)

        if cached:
            metrics.increment("cache_hits")
        else:
            metrics.increment("llm_calls")
            metrics.increment("tokens_used", random.randint(400, 600))

        if random.random() < 0.02:  # 2% error rate
            metrics.increment("errors_total")

    cache_hits = metrics.counters["cache_hits"]
    total      = metrics.counters["requests_total"]
    metrics.gauge("cache_hit_rate_pct", round(cache_hits / total * 100, 1))
    metrics.gauge("error_rate_pct",     round(metrics.counters["errors_total"] / total * 100, 1))

    metrics.report()