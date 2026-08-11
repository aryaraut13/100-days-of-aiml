# day84_job_applications/application_strategy.py
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
    max_tokens=400
)

print("[JOB APPLICATION STRATEGY — AI ENGINEER]\n")

# Generate tailored talking points for each company tier
companies = [
    {
        "company": "Sarvam AI",
        "about":   "Indian LLM company building foundation models for Indian languages",
        "why_fit": "Built RAG pipeline, LangChain agents, and fine-tuned models"
    },
    {
        "company": "Haptik",
        "about":   "Conversational AI platform for enterprise customer support",
        "why_fit": "Built ecommerce RAG support bot with FastAPI and LangSmith"
    },
    {
        "company": "Yellow.ai",
        "about":   "Enterprise AI agent platform for customer experience",
        "why_fit": "Built autonomous agents with LangGraph and 6 custom tools"
    },
]

for company in companies:
    prompt = f"""
You are helping an AI Engineer prepare a 2-sentence pitch for a job application.
Be specific and connect their experience to the company.

Company: {company['company']}
What they do: {company['about']}
Candidate's relevant experience: {company['why_fit']}

Write a 2-sentence pitch for why this candidate is a strong fit.
Be specific. No generic phrases.
"""
    print(f"[{company['company'].upper()}]")
    print(f"About: {company['about']}")
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"Pitch: {response.content}\n")
    print("-" * 55 + "\n")

print("[DAILY APPLICATION CHECKLIST]")
checklist = [
    "Read the full job description before applying",
    "Check if their tech stack matches yours (LangChain, RAG, FastAPI)",
    "Customise paragraph 1 of cover letter for this company",
    "Make sure GitHub link is in the application",
    "Set a reminder to follow up in 5 days",
]
for item in checklist:
    print(f"  [ ] {item}")