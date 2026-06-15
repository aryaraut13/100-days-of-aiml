# day29_vector_db/store.py
import chromadb
from chromadb.utils import embedding_functions

# Use ChromaDB's built-in embedding function (no API key needed for this demo)
# For production use OpenAI embeddings
ef = embedding_functions.DefaultEmbeddingFunction()

client     = chromadb.Client()
collection = client.create_collection(
    name="product_docs",
    embedding_function=ef
)

# Ecommerce FAQ documents
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

# Add to ChromaDB — it embeds automatically
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"[STORED] {collection.count()} documents in ChromaDB")
print("Collection: product_docs")
print("\nDocuments stored:")
for i, doc in enumerate(documents):
    print(f"  doc_{i}: {doc[:60]}...")