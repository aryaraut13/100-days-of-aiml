# day79_mock_interview/answers.py
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

# Practice answering common interview questions
QUESTIONS = [
    "Explain how a RAG pipeline works in under 60 seconds",
    "What is the difference between LangChain and LangGraph?",
    "How would you evaluate the quality of a RAG system?",
    "Walk me through how you would fine-tune a model with LoRA",
    "How do you handle hallucinations in production LLM applications?",
]

print("[MOCK INTERVIEW — AI ENGINEER QUESTIONS]\n")
print("Practicing 60-second answers...\n")

for i, question in enumerate(QUESTIONS, 1):
    print(f"Q{i}: {question}")

    # Generate a model answer
    response = llm.invoke([HumanMessage(content=f"""
You are an experienced AI Engineer in a job interview.
Answer this question in exactly 3-4 sentences.
Be specific, use real examples, avoid generic answers.

Question: {question}
""")])
    print(f"A:  {response.content}\n")
    print("-" * 60 + "\n")

print("[SELF EVALUATION CHECKLIST]")
criteria = [
    "Did I start with the core concept?",
    "Did I give a specific example?",
    "Did I mention production considerations?",
    "Was the answer under 60 seconds?",
    "Did I connect it to my projects?",
]
for criterion in criteria:
    print(f"  [ ] {criterion}")