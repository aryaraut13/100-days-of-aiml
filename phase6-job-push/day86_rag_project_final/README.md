# Project 1: Ecommerce RAG Support Bot — Final Version

## Problem
Customer support teams spend 60% of time answering repetitive FAQs.
This bot answers from a knowledge base — grounded, no hallucination.

## Architecture
User Query | v [Query Embedding] <- HuggingFace all-MiniLM-L6-v2 | v [ChromaDB Retrieval] <- Top 3 semantically similar chunks | v [LangChain Chain] <- LCEL pipe: retriever | prompt | llm | parser | v [Claude Haiku] <- Answer generation, grounded in context | v Grounded Answer (or "I don't have information about that")
## Stack
- LangChain — pipeline orchestration (LCEL)
- ChromaDB — persistent vector database
- HuggingFace (all-MiniLM-L6-v2) — embeddings
- Anthropic Claude Haiku — language model
- FastAPI — REST API (POST /ask)
- Streamlit — chat interface
- LangSmith — observability and tracing

## Evaluation Results
- Test questions: 5/5 answered correctly
- Average overlap score: 0.78
- Out-of-scope handling: returns "I don't have information about that"
- Average latency: 1.5s per query

## API Endpoints
- GET /health — service status
- POST /ask — submit a question
- GET /docs — Swagger UI

## Run locally
```bash
pip install -r requirements.txt
python ingest.py
streamlit run app.py
# OR
uvicorn main:app --reload
```

## What I would improve with more time
- Add conversation memory for multi-turn support
- Implement hybrid search (semantic + BM25)
- Add Redis caching for frequent queries
- Deploy on Render with CI/CD