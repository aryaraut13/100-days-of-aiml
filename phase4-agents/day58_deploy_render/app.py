import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI(
    title="Market Research Agent API",
    description="Autonomous ecommerce market intelligence",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("ANTHROPIC_API_KEY")

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=api_key,
    max_tokens=500
)

class ResearchRequest(BaseModel):
    query: str

class ResearchResponse(BaseModel):
    query: str
    result: str
    status: str

@app.get("/")
def root():
    return {
        "service": "Market Research Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set")
    try:
        response = llm.invoke([
            HumanMessage(content=f"Answer this market research question briefly: {request.query}")
        ])
        return ResearchResponse(
            query=request.query,
            result=response.content,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))