import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=300
)

# Knowledge base
FAQ_DOCS = [
    "To return a product, visit our returns portal within 30 days of purchase. You will receive a full refund within 5-7 business days.",
    "Standard shipping takes 3-5 business days. Express shipping takes 1-2 days and costs Rs.99 extra.",
    "We accept Visa, Mastercard, UPI, PayTM, and net banking. COD available for orders under Rs.2000.",
    "Track your order using the tracking number sent to your registered email within 24 hours of dispatch.",
    "All products come with a 1-year manufacturer warranty. Extended warranty available for Rs.299.",
    "Cancel your subscription anytime from Account Settings → Subscriptions → Cancel.",
    "Contact support at support@store.com or call 1800-123-4567 (Mon-Sat, 9am-6pm).",
    "Bulk orders of 10+ items receive a 15% discount.",
    "EMI options available on orders above Rs.3000. 0% EMI for 3 months.",
    "Gift wrapping costs Rs.50 per item.",
]

# Build vector store
docs       = [Document(page_content=d) for d in FAQ_DOCS]
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

print("[VECTOR STORE] Built with 10 FAQ documents")
print("[RETRIEVER]    Ready — returning top 3 results\n")


def retrieve_and_show(query: str):
    """Show what gets retrieved for a query."""
    docs = retriever.invoke(query)
    print(f"Query: {query}")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc.page_content[:80]}...")
    print()


if __name__ == "__main__":
    print("[RETRIEVAL TEST]\n")
    retrieve_and_show("How do I get my money back?")
    retrieve_and_show("What delivery options are available?")
    retrieve_and_show("Can I pay later?")