from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import QuestionRequest, AnswerResponse
from rag_bot import get_answer

app = FastAPI(
    title="Ecommerce RAG API",
    description="Customer support bot powered by RAG pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "Ecommerce RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/ask", "/health", "/docs"]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "claude-haiku-4-5-20251001"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer, sources = get_answer(request.question)
        return AnswerResponse(
            question=request.question,
            answer=answer,
            session_id=request.session_id,
            sources_used=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/questions")
def sample_questions():
    return {
        "sample_questions": [
            "How do I return a product?",
            "What payment methods do you accept?",
            "How long does shipping take?",
            "Do you offer EMI?",
            "How do I track my order?",
        ]
    }
