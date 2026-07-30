# day72_rag_advanced/hybrid_search.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

print("[HYBRID SEARCH — SEMANTIC + BM25 KEYWORD]\n")

# Knowledge base
documents = [
    "To return a product visit our returns portal within 30 days of purchase",
    "Standard shipping takes 3-5 business days free on orders above Rs.500",
    "We accept Visa Mastercard UPI PayTM and net banking at checkout",
    "Track your order using the tracking number sent to your registered email",
    "All products come with a 1-year manufacturer warranty",
    "Cancel your subscription from Account Settings anytime",
    "Contact support at support@store.com or call 1800-123-4567",
    "Bulk orders of 10 or more items receive a 15 percent discount",
    "EMI options available on orders above Rs.3000 through partner banks",
    "Gift wrapping costs Rs.50 per item with personalised message option",
]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings
doc_embeddings = model.encode(documents)

# BM25 index
tokenized_docs = [doc.lower().split() for doc in documents]
bm25           = BM25Okapi(tokenized_docs)


def semantic_search(query: str, top_k: int = 5) -> list:
    """Pure semantic search using cosine similarity."""
    query_emb = model.encode(query)
    scores    = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = float(np.dot(query_emb, doc_emb) /
                    (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)))
        scores.append((sim, i, documents[i]))
    return sorted(scores, reverse=True)[:top_k]


def bm25_search(query: str, top_k: int = 5) -> list:
    """BM25 keyword search."""
    tokenized_query = query.lower().split()
    scores          = bm25.get_scores(tokenized_query)
    ranked          = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(score, idx, documents[idx]) for idx, score in ranked[:top_k]]


def hybrid_search(query: str, top_k: int = 3,
                  semantic_weight: float = 0.7,
                  bm25_weight: float = 0.3) -> list:
    """
    Hybrid search: combine semantic and BM25 scores.
    semantic_weight + bm25_weight should equal 1.0
    """
    sem_results  = semantic_search(query, top_k=len(documents))
    bm25_results = bm25_search(query, top_k=len(documents))

    # Normalize scores to [0, 1]
    sem_scores  = {idx: score for score, idx, _ in sem_results}
    bm25_scores = {idx: score for score, idx, _ in bm25_results}

    max_sem  = max(sem_scores.values())  if sem_scores  else 1
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 1

    # Combine scores
    combined = {}
    for i in range(len(documents)):
        s = sem_scores.get(i, 0)  / max_sem
        b = bm25_scores.get(i, 0) / max_bm25
        combined[i] = semantic_weight * s + bm25_weight * b

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [(score, documents[idx]) for idx, score in ranked[:top_k]]


# Compare all three methods
queries = [
    "How do I get my money back?",
    "UPI payment checkout",
    "late delivery shipment tracking",
]

print(f"{'Method':10s} | {'Score':6s} | Result")
print("-" * 80)

for query in queries:
    print(f"\nQuery: '{query}'")
    print(f"\n  [SEMANTIC]")
    for score, _, doc in semantic_search(query, top_k=2):
        print(f"    {score:.4f} | {doc[:65]}...")

    print(f"\n  [BM25]")
    for score, _, doc in bm25_search(query, top_k=2):
        print(f"    {score:.4f} | {doc[:65]}...")

    print(f"\n  [HYBRID (70% semantic + 30% BM25)]")
    for score, doc in hybrid_search(query, top_k=2):
        print(f"    {score:.4f} | {doc[:65]}...")
    print()