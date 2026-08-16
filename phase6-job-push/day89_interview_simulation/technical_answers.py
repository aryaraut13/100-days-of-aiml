# day89_interview_simulation/technical_answers.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=400
)

print("[MOCK INTERVIEW SIMULATION — AI ENGINEER]\n")
print("=" * 60)
print("Format: Question -> My answer -> Evaluation")
print("=" * 60 + "\n")

interview_questions = [
    {
        "q": "Walk me through how you would build a customer support bot.",
        "my_answer": "I would use a RAG pipeline. First, ingest FAQs into ChromaDB with HuggingFace embeddings. Then at query time: embed the question, retrieve top 3 chunks, assemble a grounded prompt, and call Claude Haiku for generation. Add FastAPI for the API layer and LangSmith for observability. I have built exactly this — it achieves 0.78 overlap score on test questions."
    },
    {
        "q": "What is the difference between RAG and fine-tuning? When would you use each?",
        "my_answer": "RAG retrieves external documents at query time and feeds them as context. Fine-tuning bakes knowledge into model weights. Use RAG when knowledge changes frequently or you need to cite sources. Use fine-tuning when you need to change style, format, or add domain-specific vocabulary that stays constant. I have built both — RAG for the support bot, LoRA fine-tuning for intent classification with 0.44% trainable parameters."
    },
    {
        "q": "How do you handle hallucinations in production?",
        "my_answer": "Three approaches. First, RAG grounding — constrain answers to retrieved documents only, and instruct the model to say I don't have information about that for out-of-scope queries. Second, temperature zero for factual tasks removes creativity where you don't want it. Third, output validation — check that the answer references content from the retrieved chunks. I implemented all three in my RAG support bot."
    },
    {
        "q": "Design a system that can process 1 million queries per day.",
        "my_answer": "1M queries per day is about 12 per second. I would add Redis caching with 50% target hit rate, bringing effective load to 6 per second. Use Pinecone hosted vector DB for sub-20ms retrieval. Multiple FastAPI instances behind a load balancer. Message queue for traffic spikes. Monitor P95 latency with LangSmith and structured logging. The architecture layers: API Gateway, Cache Check, RAG Pipeline, Observability."
    },
]

scores = []
for i, item in enumerate(interview_questions, 1):
    print(f"Q{i}: {item['q']}")
    print(f"\nMy answer: {item['my_answer']}\n")

    # Evaluate the answer
    eval_prompt = f"""You are a senior AI Engineer conducting an interview.
Rate this answer from 1-10 and give one specific strength and one improvement.
Be concise — 3 sentences max.

Question: {item['q']}
Answer: {item['my_answer']}"""

    response = llm.invoke([HumanMessage(content=eval_prompt)])
    print(f"Evaluation: {response.content}\n")
    print("-" * 60 + "\n")

print("[INTERVIEW COMPLETE]")
print("Review evaluations above and focus on the improvement areas.")