# day71_vector_db_advanced/chroma_advanced.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import time

print("[CHROMADB ADVANCED — PERSISTENT COLLECTIONS + FILTERING]\n")

# Persistent client — data survives restarts
client = chromadb.PersistentClient(path="./chroma_store")
ef     = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create or get collection
collection = client.get_or_create_collection(
    name="ecommerce_products",
    embedding_function=ef,
    metadata={"description": "Ecommerce product catalog"}
)

# Add documents with metadata
products = [
    {
        "id":       "p001",
        "text":     "Sony WH-1000XM5 wireless noise cancelling headphones premium audio",
        "metadata": {"category": "headphones", "brand": "Sony", "price": 24990, "rating": 4.6, "in_stock": True}
    },
    {
        "id":       "p002",
        "text":     "boAt Rockerz 450 bluetooth headphones with microphone budget",
        "metadata": {"category": "headphones", "brand": "boAt", "price": 1299, "rating": 4.1, "in_stock": True}
    },
    {
        "id":       "p003",
        "text":     "JBL Tune 510BT wireless on-ear headphones mid-range",
        "metadata": {"category": "headphones", "brand": "JBL", "price": 2999, "rating": 4.2, "in_stock": True}
    },
    {
        "id":       "p004",
        "text":     "Apple AirPods Pro true wireless earbuds active noise cancellation",
        "metadata": {"category": "earbuds", "brand": "Apple", "price": 24900, "rating": 4.7, "in_stock": False}
    },
    {
        "id":       "p005",
        "text":     "Dell Inspiron 15 laptop Intel Core i5 8GB RAM office work",
        "metadata": {"category": "laptop", "brand": "Dell", "price": 45999, "rating": 4.2, "in_stock": True}
    },
    {
        "id":       "p006",
        "text":     "OnePlus Nord Buds 2 wireless earbuds noise cancellation budget",
        "metadata": {"category": "earbuds", "brand": "OnePlus", "price": 2799, "rating": 4.3, "in_stock": True}
    },
]

# Add to collection
collection.upsert(
    ids=[p["id"] for p in products],
    documents=[p["text"] for p in products],
    metadatas=[p["metadata"] for p in products]
)

print(f"Collection: {collection.name}")
print(f"Documents:  {collection.count()}\n")

# 1. Basic semantic search
print("[1. BASIC SEMANTIC SEARCH]")
query   = "wireless headphones with noise cancellation"
results = collection.query(query_texts=[query], n_results=3)
print(f"Query: '{query}'")
for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    score = round(1 / (1 + dist), 4)
    print(f"  [{score}] {meta['brand']} — Rs.{meta['price']} — {doc[:50]}...")
print()

# 2. Filtered search — only in-stock items under Rs.5000
print("[2. FILTERED SEARCH — in stock AND price < 5000]")
results = collection.query(
    query_texts=["wireless audio"],
    n_results=5,
    where={
        "$and": [
            {"in_stock": {"$eq": True}},
            {"price":    {"$lt": 5000}},
        ]
    }
)
print("Query: 'wireless audio' | Filter: in_stock=True AND price<5000")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  {meta['brand']:12s} Rs.{meta['price']:6d} | {meta['category']}")
print()

# 3. Category filter
print("[3. CATEGORY FILTER — earbuds only]")
results = collection.query(
    query_texts=["noise cancellation"],
    n_results=3,
    where={"category": {"$eq": "earbuds"}}
)
print("Query: 'noise cancellation' | Filter: category=earbuds")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  {meta['brand']:12s} Rs.{meta['price']:6d} | rating: {meta['rating']}")
print()

# 4. Performance benchmark
print("[4. PERFORMANCE BENCHMARK]")
queries = ["budget headphones", "premium noise cancellation", "laptop for office work"]
for q in queries:
    start   = time.time()
    results = collection.query(query_texts=[q], n_results=3)
    elapsed = (time.time() - start) * 1000
    print(f"  '{q[:35]}' -> {elapsed:.1f}ms")