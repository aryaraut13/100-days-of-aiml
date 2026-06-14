import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> np.ndarray:
    return model.encode(text)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_search(query: str, documents: list,
                    top_k: int = 3) -> list:
    query_emb = get_embedding(query)
    results   = []
    for doc in documents:
        doc_emb = get_embedding(doc)
        sim     = cosine_similarity(query_emb, doc_emb)
        results.append((sim, doc))
    return sorted(results, reverse=True)[:top_k]


if __name__ == "__main__":
    docs = [
        "How to return a product and get a refund",
        "Shipping times and delivery options",
        "Payment methods accepted at checkout",
        "How to track your order status",
        "Product warranty and repair policy",
        "How to cancel a subscription",
        "Customer support contact information",
    ]

    query = "I want to send back an item I bought"

    print(f"[SEMANTIC SEARCH]\n")
    print(f"Query: '{query}'\n")
    print("Top 3 most relevant documents:\n")

    results = semantic_search(query, docs, top_k=3)
    for rank, (score, doc) in enumerate(results, 1):
        print(f"  {rank}. [{score:.4f}] {doc}")

    print("\nNote: Query never used the word 'return'")
    print("But semantic search found the right document anyway.")
    print("That's meaning over keywords.")

