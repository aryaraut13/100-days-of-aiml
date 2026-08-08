# day81_linkedin_optimization/profile_audit.py
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
    max_tokens=500
)

print("[LINKEDIN PROFILE AUDIT — AI ENGINEER ROLE]\n")

# Current profile sections to audit
profile = {
    "headline":   "AI Engineer | LangChain . RAG . LLM Applications | Building...",
    "about":      "Co-founder building AI products. 81 days into a 100-day AI/ML challenge posting daily.",
    "experience": "Co-founder at startup. Previously student.",
    "skills":     ["Python", "Machine Learning", "LangChain", "RAG", "FastAPI"],
    "projects":   ["RAG Support Bot", "Market Research Agent", "Product Research Agent"],
}

# Audit each section
sections = [
    ("Headline", profile["headline"],
     "Audit this LinkedIn headline for an AI Engineer job seeker. Is it optimized for recruiters? Suggest improvements in 2 sentences."),
    ("About section", profile["about"],
     "Audit this LinkedIn about section for AI Engineer role. What is missing? Suggest 2 specific improvements."),
    ("Skills", str(profile["skills"]),
     "Audit these LinkedIn skills for an AI Engineer role. What important skills are missing? List 5 missing skills."),
]

for section, content, prompt in sections:
    print(f"[{section.upper()}]")
    print(f"Current: {content}")
    response = llm.invoke([HumanMessage(content=f"{prompt}\nCurrent content: {content}")])
    print(f"Audit:   {response.content}\n")
    print("-" * 60 + "\n")

# Optimized versions
print("[OPTIMIZED SECTIONS]\n")

optimizations = [
    ("Headline", "AI Engineer | LangChain . RAG . LLM Agents | 3 Production AI Apps Shipped | 81 Days Building in Public"),
    ("Skills to add", "LangGraph, HuggingFace, Docker, LoRA Fine-tuning, Prompt Engineering, Vector Databases, FastAPI, LangSmith"),
    ("Featured projects", "Pin Project 1 (RAG Bot), Project 2 (Market Research Agent), GitHub repo"),
]

for section, content in optimizations:
    print(f"{section}:")
    print(f"  {content}\n")

print("[RECRUITER SEARCH KEYWORDS TO ADD]")
keywords = [
    "LLM Engineer", "AI Engineer", "LangChain", "RAG",
    "Generative AI", "LangGraph", "Vector Database",
    "Fine-tuning", "Prompt Engineering", "AI Agents"
]
print("  " + " | ".join(keywords))