import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from auth import verify_api_key, check_rate_limit

load_dotenv()

app = FastAPI(
    title="Secure Market Research API",
    description="API key authenticated + rate limited",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=300
)


class ResearchRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"service": "Secure Market Research API", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/research")
def research(
    request: ResearchRequest,
    api_key: str = Depends(verify_api_key)
):
    # Rate limit check
    check_rate_limit(api_key)

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    response = llm.invoke([
        HumanMessage(content=f"Answer briefly: {request.query}")
    ])
    return {
        "query":     request.query,
        "result":    response.content,
        "status":    "success",
        "api_key":   f"{api_key[:8]}..."
    }


if __name__ == "__main__":
    import uvicorn
    print("[SECURE API]\n")
    print("Test with valid key:")
    print('Invoke-RestMethod -Uri "http://localhost:8001/research" -Method POST -Headers @{"X-API-Key"="dev-key-123"} -ContentType "application/json" -Body \'{"query": "headphones market"}\'\n')
    print("Test without key:")
    print('Invoke-RestMethod -Uri "http://localhost:8001/research" -Method POST -ContentType "application/json" -Body \'{"query": "test"}\'\n')
    uvicorn.run(app, host="0.0.0.0", port=8001)