import os
import chromadb
from anthropic import Anthropic
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
chroma_client    = chromadb.Client()
ef               = SentenceTransformerEmbeddingFunction(
                       model_name="all-MiniLM-L6-v2"
                   )

collection = chroma_client.create_collection(
    name="rag_docs",
    embedding_function=ef
)

DOCUMENTS = [
    "To return a product, visit our returns portal within 30 days of purchase. You will receive a full refund within 5-7 business days.",
    "Standard shipping takes 3-5 business days. Express shipping takes 1-2 days and costs Rs.99 extra.",
    "We accept Visa, Mastercard, UPI, PayTM, and net banking. COD is available for orders under Rs.2000.",
    "Track your order using the tracking number sent to your registered email within 24 hours of dispatch.",
    "All products come with a 1-year manufacturer warranty. Extended warranty of 2 years available for Rs.299.",
    "Cancel your subscription anytime from Account Settings → Subscriptions → Cancel.",
    "Contact our support team at support@store.com or call 1800-123-4567 (Mon-Sat, 9am-6pm).",
    "Bulk orders of 10+ items receive a 15% discount. Contact our B2B team for orders above Rs.50,000.",
    "EMI options available on orders above Rs.3000 through partner banks. 0% EMI for 3 months.",
    "Gift wrapping costs Rs.50 per item. Add a personalised message at no extra charge.",
]

collection.add(
    documents=DOCUMENTS,
    ids=[f"doc_{i}" for i in range(len(DOCUMENTS))]
)


def retrieve(query: str, top_k: int = 3) -> list[str]:
    """Retrieve most relevant documents from ChromaDB."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents"]
    )
    return results["documents"][0]


def generate(query: str, context_docs: list[str]) -> str:
    """Generate answer using retrieved context."""
    context = "\n".join([f"- {doc}" for doc in context_docs])

    system_prompt = """You are a helpful customer support assistant for an ecommerce store.
Answer questions ONLY using the provided context.
If the answer is not in the context, say exactly: "I don't have information about that."
Be concise and friendly. Answer in 2-3 sentences max."""

    user_message = f"""Context:
{context}

Customer question: {query}"""

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text


def rag_pipeline(query: str) -> str:
    """Full RAG pipeline: retrieve → generate."""
    print(f"\n[QUERY]    {query}")

    docs = retrieve(query, top_k=3)
    print(f"[RETRIEVE] {len(docs)} documents found")
    for i, doc in enumerate(docs, 1):
        print(f"           {i}. {doc[:65]}...")

    answer = generate(query, docs)
    print(f"[GENERATE] {answer}")
    return answer