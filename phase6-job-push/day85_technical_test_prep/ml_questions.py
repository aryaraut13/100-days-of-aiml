# day85_technical_test_prep/ml_questions.py
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

print("[ML TECHNICAL QUESTIONS — SELF TEST]\n")

questions = [
    "What is the difference between precision and recall? Give an example where recall matters more.",
    "Explain gradient descent in 3 sentences. What is the learning rate?",
    "What is overfitting? Name 3 ways to prevent it.",
    "What is the bias-variance tradeoff?",
    "When would you use Random Forest over XGBoost?",
]

for i, question in enumerate(questions, 1):
    print(f"Q{i}: {question}")
    response = llm.invoke([HumanMessage(content=f"Answer this ML interview question concisely in 2-3 sentences with a specific example:\n{question}")])
    print(f"A:  {response.content}\n")
    print("-" * 60 + "\n")