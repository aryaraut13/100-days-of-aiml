import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=300
)

FAQ_DOCS = [
    "To return a product, visit our returns portal within 30 days of purchase. You will receive a full refund within 5-7 business days.",
    "Standard shipping takes 3-5 business days. Express shipping takes 1-2 days and costs Rs.99 extra.",
    "We accept Visa, Mastercard, UPI, PayTM, and net banking. COD available for orders under Rs.2000.",
    "Track your order using the tracking number sent to your registered email within 24 hours of dispatch.",
    "All products come with a 1-year manufacturer warranty. Extended warranty available for Rs.299.",
    "Cancel your subscription anytime from Account Settings -> Subscriptions -> Cancel.",
    "Contact support at support@store.com or call 1800-123-4567 (Mon-Sat, 9am-6pm).",
    "Bulk orders of 10+ items receive a 15% discount.",
    "EMI options available on orders above Rs.3000. 0% EMI for 3 months.",
    "Gift wrapping costs Rs.50 per item.",
]

docs        = [Document(page_content=d) for d in FAQ_DOCS]
embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer support assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say "I don't have information about that."
Be concise - 2-3 sentences max.

Context:
{context}"""),
    ("human", "{question}")
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def ask(question: str) -> str:
    return chain.invoke(question)


if __name__ == "__main__":
    print("[LANGCHAIN RAG CHAIN]\n")
    questions = [
        "How do I return a product?",
        "What payment methods are available?",
        "Do you have EMI options?",
        "What are your office hours?",
    ]

    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask(q)}\n")
        print("-" * 55)