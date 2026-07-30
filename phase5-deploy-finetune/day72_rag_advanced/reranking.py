# day72_rag_advanced/reranking.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

print("[RERANKING — TWO-STAGE RETRIEVAL]\n")

# Stage 1: Fast retrieval with bi-encoder
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Stage 2: Accurate reranking with cross-encoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

documents = [
    "To return a product visit our returns portal within 30 days",
    "Standard shipping takes 3-5 business days",
    "We accept Visa Mastercard UPI PayTM and net banking",
    "Track your order using the tracking number sent to your email",
    "All products come with a 1-year manufacturer warranty",
    "Refund will be processed within 5-7 business days after return",
    "Return policy: items must be unused and in original packaging",
    "For damaged items contact support within 48 hours of delivery",
]

doc_embeddings = bi_encoder.encode(documents)


def two_stage_retrieval(query: str, first_stage_k: int = 5, final_k: int = 3) -> list:
    """
    Stage 1: Fast retrieval with bi-encoder (top 5)
    Stage 2: Accurate reranking with cross-encoder (top 3)
    """
    # Stage 1: Bi-encoder retrieval
    query_emb    = bi_encoder.encode(query)
    scores       = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = float(np.dot(query_emb, doc_emb) /
                    (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)))
        scores.append((sim, i, documents[i]))
    stage1 = sorted(scores, reverse=True)[:first_stage_k]

    # Stage 2: Cross-encoder reranking
    pairs        = [[query, doc] for _, _, doc in stage1]
    cross_scores = cross_encoder.predict(pairs)
    reranked     = sorted(
        zip(cross_scores, [doc for _, _, doc in stage1]),
        reverse=True
    )
    return reranked[:final_k]


queries = [
    "I want to return my damaged product",
    "How long does refund take?",
]

for query in queries:
    print(f"Query: '{query}'")
    results = two_stage_retrieval(query)
    for rank, (score, doc) in enumerate(results, 1):
        print(f"  {rank}. [{score:.4f}] {doc}")
    print()