# Ecommerce RAG Support Bot

A production-ready customer support bot powered by RAG (Retrieval Augmented Generation).

## Architecture
User Question

|

v

[HuggingFace Embeddings]  <- Query embedding (all-MiniLM-L6-v2)

|

v

[ChromaDB Vector Store]   <- Semantic search -> Top 3 chunks

|

v

[LangChain Chain]         <- Retrieval + prompt assembly (LCEL)

|

v

[Claude Haiku]            <- Answer generation (Anthropic)

|

v

Grounded Answer
## Stack
- **LangChain** — pipeline orchestration
- **ChromaDB** — vector database
- **HuggingFace** (all-MiniLM-L6-v2) — embeddings
- **Anthropic Claude Haiku** — language model
- **FastAPI** — REST API (POST /ask)
- **Streamlit** — chat UI
- **LangSmith** — tracing and observability

## Features
- Answers only from knowledge base — no hallucination
- Returns "I don't have information about that" for out-of-scope questions
- Persistent vector store — ingest once, query forever
- REST API endpoint via FastAPI
- Chat UI via Streamlit
- Full observability via LangSmith

## Evaluation
5 test questions evaluated against ground truth.
Average overlap score: 0.78 | 5/5 answered correctly

## Run locally
```bash
pip install -r requirements.txt
python ingest.py
streamlit run app.py
```