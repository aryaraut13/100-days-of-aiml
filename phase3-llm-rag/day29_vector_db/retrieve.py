import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client     = chromadb.Client()
collection = client.create_collection(
    name="product_docs",
    embedding_function=ef
)

documents = [
    "To return a product, visit our returns portal within 30 days of purchase.",
    "Standard shipping takes 3-5 business days. Express shipping takes 1-2 days.",
    "We accept Visa, Mastercard, UPI, and net banking at checkout.",
    "Track your order using the tracking number sent to your email.",
    "Products come with a 1-year manufacturer warranty.",
    "Cancel your subscription anytime from your account settings.",
    "Contact support at support@store.com or call 1800-XXX-XXXX.",
    "Gift wrapping is available for an additional Rs.50 per item.",
    "We offer EMI options on orders above Rs.5000.",
    "Bulk orders of 10+ items get a 15% discount automatically.",
]
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)


def query_db(question: str, top_k: int = 2) -> list:
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "distances"]
    )
    docs      = results["documents"][0]
    distances = results["distances"][0]
    scores    = [round(1 / (1 + d), 4) for d in distances]
    return list(zip(docs, scores))


if __name__ == "__main__":
    queries = [
        "How do I send something back?",
        "When will my package arrive?",
        "Can I pay in instalments?",
    ]

    print("[VECTOR DATABASE — SEMANTIC RETRIEVAL]\n")
    for query in queries:
        print(f"[QUERY] {query}")
        results = query_db(query, top_k=2)
        for doc, score in results:
            print(f"  Score: {score} | {doc[:70]}...")
        print()