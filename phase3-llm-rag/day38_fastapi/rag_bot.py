# day38_fastapi/rag_bot.py
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "day35_project1_rag_bot", "chroma_db"
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain():
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=300
    )

    embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful customer support assistant.
Answer ONLY using the provided context.
If not in context, say "I don't have information about that."
Be concise - 2-3 sentences max.

Context: {context}"""),
        ("human", "{question}")
    ])

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    ), retriever


chain, retriever = build_chain()


def get_answer(question: str) -> tuple[str, int]:
    """Returns (answer, number_of_source_docs)."""
    docs   = retriever.invoke(question)
    answer = chain.invoke(question)
    return answer, len(docs)