import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# LangSmith tracing is enabled automatically via env vars
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_key
# LANGCHAIN_PROJECT=100-days-rag

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=300
)

CHROMA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "day35_project1_rag_bot", "chroma_db"
)

embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer support assistant.
Answer ONLY using the provided context.
If not in context say "I don't have information about that."
Be concise.

Context: {context}"""),
    ("human", "{question}")
])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print("[LANGSMITH TRACED CHAIN]\n")
    print("All runs are being logged to LangSmith.")
    print("Check: https://smith.langchain.com\n")

    questions = [
        "How do I return a product?",
        "What payment methods do you accept?",
        "What are your store hours?",  # not in KB — will show in traces
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = chain.invoke(q)
        print(f"A: {answer}\n")

    print("Check LangSmith dashboard for full traces,")
    print("token usage, latency, and retrieval details.")