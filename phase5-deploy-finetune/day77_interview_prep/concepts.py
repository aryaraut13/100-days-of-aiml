# day77_interview_prep/concepts.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=300
)

# Key concepts to master for AI Engineer interviews
CONCEPTS = [
    "RAG pipeline components and flow",
    "Difference between fine-tuning and RAG",
    "ReAct agent pattern",
    "Vector database use cases",
    "LLM evaluation metrics",
    "LoRA and parameter-efficient fine-tuning",
    "Chunking strategies for RAG",
    "Prompt engineering techniques",
]

print("[AI ENGINEER INTERVIEW CONCEPTS — SELF TEST]\n")
print("Testing understanding of core concepts...\n")

for concept in CONCEPTS[:4]:  # Test first 4 to save API calls
    print(f"Concept: {concept}")
    response = llm.invoke([HumanMessage(content=f"Explain '{concept}' in exactly 2 sentences for an AI engineer interview.")])
    print(f"Answer:  {response.content}\n")
    print("-" * 60 + "\n")

print("[INTERVIEW CHECKLIST]")
checklist = [
    ("RAG pipeline",          True),
    ("LangChain chains",      True),
    ("LangGraph agents",      True),
    ("ChromaDB + embeddings", True),
    ("FastAPI deployment",    True),
    ("Docker basics",         True),
    ("LoRA fine-tuning",      True),
    ("System design",         True),
    ("Evaluation metrics",    True),
    ("Streaming + caching",   True),
]

done  = sum(1 for _, v in checklist if v)
total = len(checklist)
print(f"\nCompleted: {done}/{total} topics")
for topic, done_flag in checklist:
    status = "DONE" if done_flag else "TODO"
    print(f"  {status:5s} | {topic}")