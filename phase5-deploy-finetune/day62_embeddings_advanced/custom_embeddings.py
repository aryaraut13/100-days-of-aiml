import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import time
from sentence_transformers import SentenceTransformer

print("[BATCH EMBEDDING — PRODUCTION APPROACH]\n")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Simulate a product catalog
products = [
    "Sony WH-1000XM5 wireless noise cancelling headphones",
    "boAt Rockerz 450 bluetooth headphones with mic",
    "JBL Tune 510BT wireless on-ear headphones",
    "Apple AirPods Pro with active noise cancellation",
    "Samsung Galaxy Buds2 true wireless earbuds",
    "OnePlus Nord Buds 2 with adaptive noise cancellation",
    "Dell Inspiron 15 laptop with Intel Core i5",
    "HP Pavilion 14 laptop 8GB RAM 512GB SSD",
    "Lenovo IdeaPad Gaming laptop with RTX 3050",
    "Apple MacBook Air M2 chip 13 inch",
]

print(f"Embedding {len(products)} products...")
start = time.time()
embeddings = model.encode(products, batch_size=8, show_progress_bar=False)
elapsed = time.time() - start

print(f"Done in {elapsed:.3f}s | Shape: {embeddings.shape}\n")


def semantic_search(query: str, top_k: int = 3) -> list:
    query_emb = model.encode(query)
    scores    = []
    for i, emb in enumerate(embeddings):
        sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb)))
        scores.append((sim, products[i]))
    return sorted(scores, reverse=True)[:top_k]


queries = [
    "noise cancelling wireless headphones",
    "budget gaming laptop",
    "apple products",
]

print("[SEMANTIC SEARCH RESULTS]")
for query in queries:
    print(f"\nQuery: '{query}'")
    results = semantic_search(query, top_k=3)
    for rank, (score, product) in enumerate(results, 1):
        print(f"  {rank}. [{score:.4f}] {product}")