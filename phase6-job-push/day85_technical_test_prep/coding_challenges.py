# day85_technical_test_prep/coding_challenges.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("[TECHNICAL TEST PREP — CODING CHALLENGES]\n")
print("Common patterns in AI Engineer technical tests:\n")

# Challenge 1: Implement cosine similarity
def cosine_similarity(a: list, b: list) -> float:
    """Implement cosine similarity without numpy."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a      = sum(x ** 2 for x in a) ** 0.5
    norm_b      = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


print("[CHALLENGE 1: Cosine Similarity from scratch]")
v1 = [1, 2, 3, 4, 5]
v2 = [5, 4, 3, 2, 1]
v3 = [1, 2, 3, 4, 5]  # identical to v1
result1 = cosine_similarity(v1, v2)
result2 = cosine_similarity(v1, v3)
print(f"v1 vs v2 (different): {result1:.4f}")
print(f"v1 vs v3 (identical): {result2:.4f}")
print(f"Expected: < 1.0 and 1.0 respectively")
print(f"PASS\n" if result2 == 1.0 else "FAIL\n")


# Challenge 2: Implement simple TF-IDF
def tf_idf(documents: list, query_term: str) -> list:
    """
    Compute TF-IDF score for a term across documents.
    Returns (document_index, score) tuples sorted by score.
    """
    import math
    n_docs = len(documents)

    # IDF: log(N / df) where df = number of docs containing term
    df  = sum(1 for doc in documents if query_term.lower() in doc.lower().split())
    idf = math.log(n_docs / (df + 1))  # +1 to avoid division by zero

    scores = []
    for i, doc in enumerate(documents):
        words = doc.lower().split()
        tf    = words.count(query_term.lower()) / len(words)
        score = tf * idf
        scores.append((i, round(score, 4)))

    return sorted(scores, key=lambda x: x[1], reverse=True)


print("[CHALLENGE 2: TF-IDF from scratch]")
docs = [
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks for machine learning tasks",
    "natural language processing is used in chatbots",
    "machine learning models require training data",
]
results = tf_idf(docs, "machine")
print(f"TF-IDF scores for 'machine':")
for idx, score in results:
    print(f"  Doc {idx}: {score:.4f} | {docs[idx][:50]}...")
print()


# Challenge 3: Implement a simple RAG retriever
def simple_retriever(query: str, documents: list, top_k: int = 2) -> list:
    """
    Simple keyword-based retriever (no embeddings needed for demo).
    Returns top_k most relevant documents.
    """
    query_words = set(query.lower().split())
    scores      = []

    for i, doc in enumerate(documents):
        doc_words = set(doc.lower().split())
        overlap   = len(query_words & doc_words)
        scores.append((overlap, i, doc))

    scores.sort(reverse=True)
    return [(doc, score) for score, _, doc in scores[:top_k]]


print("[CHALLENGE 3: Simple RAG Retriever]")
knowledge_base = [
    "Returns must be initiated within 30 days of purchase",
    "Shipping takes 3-5 business days for standard delivery",
    "We accept UPI, Visa, Mastercard for payment",
    "Track your order with the number sent to your email",
    "Warranty claims require proof of purchase",
]

query   = "How do I return my order?"
results = simple_retriever(query, knowledge_base)
print(f"Query: '{query}'")
for doc, score in results:
    print(f"  Score {score}: {doc}")
print()

print("[SUMMARY]")
print("Challenge 1 (Cosine Similarity): DONE")
print("Challenge 2 (TF-IDF):            DONE")
print("Challenge 3 (RAG Retriever):      DONE")
print("\nKey insight: Interviewers want to see you understand")
print("the math behind the tools you use daily.")