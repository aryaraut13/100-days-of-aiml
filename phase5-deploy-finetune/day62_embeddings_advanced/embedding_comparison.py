import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import time
from sentence_transformers import SentenceTransformer

print("[EMBEDDING MODEL COMPARISON]\n")

models = {
    "all-MiniLM-L6-v2":      "Fastest, smallest (22M params)",
    "all-mpnet-base-v2":     "Best quality, slower (110M params)",
    "paraphrase-MiniLM-L3-v2": "Ultra-fast, lower quality (17M params)",
}

test_pairs = [
    ("I want to return my product",          "How do I get a refund?"),
    ("The laptop battery is dead",           "My computer won't charge"),
    ("I love this headphone",                "The weather is sunny today"),
    ("Payment failed at checkout",           "My transaction was declined"),
]


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


for model_name, description in models.items():
    print(f"Model: {model_name}")
    print(f"Desc:  {description}")

    start  = time.time()
    model  = SentenceTransformer(model_name)
    elapsed = time.time() - start
    print(f"Load time: {elapsed:.2f}s | Dims: {model.get_sentence_embedding_dimension()}")

    print("Similarity scores:")
    for s1, s2 in test_pairs:
        e1  = model.encode(s1)
        e2  = model.encode(s2)
        sim = cosine_sim(e1, e2)
        bar = "#" * int(sim * 15)
        print(f"  {sim:.4f} {bar} | '{s1[:35]}' vs '{s2[:35]}'")
    print()