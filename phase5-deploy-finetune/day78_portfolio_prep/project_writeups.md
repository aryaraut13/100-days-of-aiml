# Project Writeups for Portfolio — Day 78

## Project 1: Ecommerce RAG Support Bot

**Problem:** Customer support teams spend 60% of time answering repetitive FAQs.
**Solution:** RAG pipeline that answers from a knowledge base — grounded, no hallucination.
**Stack:** LangChain + ChromaDB + Claude + FastAPI + Streamlit + LangSmith
**Key metrics:** 5/5 test questions answered correctly. Avg overlap score: 0.78.
**Differentiator:** Grounded responses only. "I don't have information" for out-of-scope.
**GitHub:** phase3-llm-rag/day35_project1_rag_bot/

## Project 2: Ecommerce Market Research Agent

**Problem:** Market research takes analysts days. Structured data is scattered.
**Solution:** Autonomous agent that searches, analyses, and writes reports in seconds.
**Stack:** LangChain Agents + LangGraph + Claude + Streamlit
**Tools:** 6 custom tools (product search, price segments, trends, reviews, competitor, report)
**Differentiator:** Full autonomous workflow — no human in the loop after query submission.
**GitHub:** phase4-agents/day49_project2_ui/

## Project 3: Product Research Agent

**Problem:** Buyers spend hours comparing products across multiple sites.
**Solution:** Agent that researches, compares, and recommends products instantly.
**Stack:** LangChain Agents + Claude + Streamlit
**Tools:** 4 tools (search, reviews, compare, buying guide)
**Differentiator:** Personalised recommendations by budget and use case.
**GitHub:** phase5-deploy-finetune/day76_project3_build/

## How to present projects in interviews

1. Start with the problem — not the tech
2. Explain what makes it production-ready (not just a demo)
3. Mention the specific metrics (accuracy, latency, cost)
4. Talk about what you would improve with more time
5. Have the GitHub link ready — interviewers will check it