# day71_vector_db_advanced/pinecone_basics.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from sentence_transformers import SentenceTransformer

print("[PINECONE BASICS — CONCEPT DEMO]\n")
print("Note: Using local simulation since Pinecone requires a paid account.")
print("The API pattern is identical — only the client changes.\n")

# Simulated Pinecone-style interface
class MockPineconeIndex:
    """
    Simulates Pinecone API locally.
    Real Pinecone code would be:

    import pinecone
    pc    = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("ecommerce-products")
    index.upsert(vectors=[(id, embedding, metadata)])
    results = index.query(vector=query_embedding, top_k=3, filter={...})
    """

    def __init__(self, name: str, dimension: int):
        self.name      = name
        self.dimension = dimension
        self._store    = {}
        print(f"Index created: {name} | dimension: {dimension}")

    def upsert(self, vectors: list) -> dict:
        """vectors: list of (id, embedding, metadata)"""
        for vec_id, embedding, metadata in vectors:
            self._store[vec_id] = {
                "id":        vec_id,
                "values":    embedding,
                "metadata":  metadata
            }
        return {"upserted_count": len(vectors)}

    def query(self, vector: np.ndarray, top_k: int = 3,
              filter: dict = None) -> dict:
        """Query by vector similarity with optional metadata filter."""
        scores = []
        for vec_id, entry in self._store.items():
            # Apply metadata filter
            if filter:
                match = all(
                    entry["metadata"].get(k) == v
                    for k, v in filter.items()
                )
                if not match:
                    continue

            # Cosine similarity
            a   = np.array(entry["values"])
            b   = vector
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            scores.append({
                "id":       vec_id,
                "score":    round(sim, 4),
                "metadata": entry["metadata"]
            })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return {"matches": scores[:top_k]}

    def describe_index_stats(self) -> dict:
        return {
            "index_name":       self.name,
            "dimension":        self.dimension,
            "total_vector_count": len(self._store),
            "namespaces":       {"default": len(self._store)}
        }

    def delete(self, ids: list) -> dict:
        for vec_id in ids:
            self._store.pop(vec_id, None)
        return {"deleted": len(ids)}


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
DIM   = model.get_sentence_embedding_dimension()

# Create index
index = MockPineconeIndex("ecommerce-products", dimension=DIM)
print()

# Products to index
products = [
    ("p1", "Sony WH-1000XM5 wireless noise cancelling headphones premium",
     {"category": "headphones", "brand": "Sony",   "price": 24990, "in_stock": True}),
    ("p2", "boAt Rockerz 450 budget bluetooth headphones with mic",
     {"category": "headphones", "brand": "boAt",   "price": 1299,  "in_stock": True}),
    ("p3", "JBL Tune 510BT wireless on-ear mid-range headphones",
     {"category": "headphones", "brand": "JBL",    "price": 2999,  "in_stock": True}),
    ("p4", "Apple AirPods Pro true wireless earbuds noise cancellation",
     {"category": "earbuds",    "brand": "Apple",  "price": 24900, "in_stock": False}),
    ("p5", "Dell Inspiron 15 laptop Intel Core i5 office work",
     {"category": "laptop",     "brand": "Dell",   "price": 45999, "in_stock": True}),
    ("p6", "OnePlus Nord Buds 2 budget earbuds noise cancellation",
     {"category": "earbuds",    "brand": "OnePlus","price": 2799,  "in_stock": True}),
]

# Upsert with embeddings
print("[UPSERT]")
vectors = [
    (pid, model.encode(text).tolist(), meta)
    for pid, text, meta in products
]
result = index.upsert(vectors)
print(f"Upserted: {result['upserted_count']} vectors")

# Stats
stats = index.describe_index_stats()
print(f"\n[INDEX STATS]")
print(f"Index name:   {stats['index_name']}")
print(f"Dimension:    {stats['dimension']}")
print(f"Total vectors:{stats['total_vector_count']}")

# Basic query
print(f"\n[BASIC QUERY]")
query    = "wireless headphones with good battery"
query_emb = model.encode(query)
results  = index.query(vector=query_emb, top_k=3)
print(f"Query: '{query}'")
for match in results["matches"]:
    m = match["metadata"]
    print(f"  [{match['score']}] {m['brand']:10s} Rs.{m['price']:6d} | {m['category']}")

# Filtered query
print(f"\n[FILTERED QUERY — earbuds in stock only]")
results = index.query(
    vector=model.encode("noise cancellation wireless"),
    top_k=3,
    filter={"category": "earbuds", "in_stock": True}
)
for match in results["matches"]:
    m = match["metadata"]
    print(f"  [{match['score']}] {m['brand']:10s} Rs.{m['price']:6d} | in_stock: {m['in_stock']}")

# Delete demo
print(f"\n[DELETE]")
result = index.delete(ids=["p4"])
print(f"Deleted {result['deleted']} vector(s)")
print(f"Vectors remaining: {index.describe_index_stats()['total_vector_count']}")

print(f"\n[REAL PINECONE CODE PATTERN]")
print("""
import pinecone
pc    = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("ecommerce-products")

# Upsert
index.upsert(vectors=[(id, embedding.tolist(), metadata)])

# Query
results = index.query(
    vector=query_embedding.tolist(),
    top_k=3,
    filter={"category": "earbuds"}
)

# Delete
index.delete(ids=["p4"])
""")