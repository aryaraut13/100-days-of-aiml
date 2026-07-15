import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks
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

# langchain_core only imports BaseCache/Callbacks under TYPE_CHECKING,
# so pydantic can't resolve ChatAnthropic's forward refs on its own.
# Explicit rebuild forces it to resolve using this module's namespace.
ChatAnthropic.model_rebuild()

llm = None
if api_key:
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
    if llm is None:
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
