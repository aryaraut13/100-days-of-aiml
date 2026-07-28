import sys
sys.stdout.reconfigure(encoding='utf-8')

import logging
import json
import time
from datetime import datetime
from pathlib import Path


class StructuredLogger:
    """JSON structured logger for AI applications."""

    def __init__(self, name: str, log_file: str = "app.log"):
        self.logger   = logging.getLogger(name)
        self.log_file = log_file

        if not self.logger.handlers:
            # Console handler
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            self.logger.addHandler(console)

            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

        self.logger.setLevel(logging.DEBUG)

    def _log(self, level: str, event: str, **kwargs):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level":     level,
            "event":     event,
            **kwargs
        }
        line = json.dumps(record)
        getattr(self.logger, level.lower())(line)

    def request(self, query: str, category: str, latency_ms: float, tokens: int, cached: bool):
        self._log("INFO", "llm_request",
                  query=query[:50],
                  category=category,
                  latency_ms=round(latency_ms, 1),
                  tokens=tokens,
                  cached=cached)

    def error(self, event: str, error: str, query: str = ""):
        self._log("ERROR", event, error=error, query=query[:50])

    def metric(self, name: str, value: float, unit: str = ""):
        self._log("DEBUG", "metric", metric=name, value=value, unit=unit)


if __name__ == "__main__":
    print("[STRUCTURED LOGGING DEMO]\n")

    logger = StructuredLogger("ai_app", "demo.log")

    # Simulate requests
    requests = [
        ("How do I return a product?", "returns", 1523.4, 487, False),
        ("Where is my order?",         "shipping", 1201.7, 432, False),
        ("How do I return a product?", "returns",    48.2,   0, True),
        ("Payment failed",             "payment",  1678.9, 521, False),
        ("Warranty claim",             "warranty", 1334.5, 468, False),
    ]

    for query, cat, latency, tokens, cached in requests:
        logger.request(query, cat, latency, tokens, cached)

    logger.metric("cache_hit_rate", 0.20, "%")
    logger.metric("avg_latency",    1434.6, "ms")
    logger.metric("total_tokens",   1908, "tokens")

    print("\nLogs written to demo.log")
    print("Reading structured logs:\n")

    with open("demo.log") as f:
        for line in f:
            record = json.loads(line.strip())
            if record["level"] == "INFO":
                print(f"  [{record['timestamp'][:19]}] {record['event']}")
                print(f"    query={record.get('query', '')[:40]} | latency={record.get('latency_ms', '')}ms | cached={record.get('cached', '')}")