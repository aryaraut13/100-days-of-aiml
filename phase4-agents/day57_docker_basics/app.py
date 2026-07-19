import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="Market Research Agent API",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    result: str
    status: str


@app.get("/")
def root():
    return {
        "service": "Market Research Agent",
        "version": "1.0.0",
        "status":  "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model":  "claude-haiku-4-5-20251001",
        "env":    "production" if os.getenv("ENV") == "prod" else "development"
    }


@app.post("/research", response_model=QueryResponse)
def research(request: QueryRequest):
    # Simplified for Docker demo
    return QueryResponse(
        query=request.query,
        result=f"Research completed for: {request.query}",
        status="success"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)