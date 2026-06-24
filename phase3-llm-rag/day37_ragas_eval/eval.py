# day37_ragas_eval/eval.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from test_questions import TEST_DATA

load_dotenv()

# Build the RAG chain (same as day35)
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=300
)

embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the persisted vectorstore from day35
CHROMA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "day35_project1_rag_bot", "chroma_db"
)

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
If not in context, say "I don't have information about that."
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


def simple_eval():
    """
    Manual evaluation — checks if ground truth keywords appear in answer.
    Simple but effective for quick assessment.
    """
    print("[RAG EVALUATION]\n")
    print(f"{'Question':45s} {'Match':6s} {'Score':6s}")
    print("-" * 65)

    scores = []
    for item in TEST_DATA:
        question     = item["question"]
        ground_truth = item["ground_truth"].lower()
        answer       = chain.invoke(question).lower()

        # Check key word overlap
        gt_words     = set(ground_truth.split())
        ans_words    = set(answer.split())
        overlap      = len(gt_words & ans_words) / len(gt_words)
        match        = "✓" if overlap > 0.3 else "✗"
        scores.append(overlap)

        print(f"{question[:44]:45s} {match:6s} {overlap:.2f}")

    print("-" * 65)
    print(f"\n[RESULTS]")
    print(f"  Average overlap score: {sum(scores)/len(scores):.2f}")
    print(f"  Questions answered:    {sum(1 for s in scores if s > 0.3)}/{len(scores)}")
    print(f"\n[DETAILED ANSWERS]\n")

    for item in TEST_DATA:
        print(f"Q: {item['question']}")
        print(f"A: {chain.invoke(item['question'])}")
        print()


if __name__ == "__main__":
    simple_eval()

