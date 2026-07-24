# day67_fastapi_advanced/main.py
import os
import time
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from models import AIRequest, AIResponse, BatchRequest
from middleware import LoggingMiddleware, RateLimitMiddleware

load_dotenv()

app = FastAPI(title="Advanced AI API", version="2.0.0")

# Add middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=50, window=60)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=300
)

# Background task — log to file
def log_request(task: str, text: str, result: str):
    with open("requests.log", "a") as f:
        f.write(f"{time.time():.0f}|{task}|{len(text)}chars|{result[:50]}\n")


@app.get("/")
async def root():
    return {"service": "Advanced AI API", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "claude-haiku-4-5-20251001"}


@app.post("/ai", response_model=AIResponse)
async def process(request: AIRequest, background_tasks: BackgroundTasks):
    start = time.time()

    prompts = {
        "summarize": f"Summarize in 3 bullet points using -> arrows:\n{request.text}",
        "classify":  f"Classify as positive/negative/neutral. One word only:\n{request.text}",
        "chat":      request.text,
    }

    prompt   = prompts.get(request.task, request.text)
    response = llm.invoke([HumanMessage(content=prompt)])
    result   = response.content
    duration = (time.time() - start) * 1000

    # Log in background — doesn't slow down response
    background_tasks.add_task(log_request, request.task, request.text, result)

    return AIResponse(
        task=request.task,
        input_text=request.text[:100],
        result=result,
        process_time_ms=round(duration, 1)
    )


@app.post("/batch")
async def batch_process(request: BatchRequest):
    """Process multiple texts concurrently."""
    async def process_one(text: str) -> str:
        response = llm.invoke([HumanMessage(content=f"Classify as positive/negative/neutral. One word:\n{text}")])
        return response.content.strip()

    tasks   = [process_one(text) for text in request.texts]
    results = await asyncio.gather(*tasks)

    return {
        "task":    request.task,
        "results": [{"text": t[:50], "result": r} for t, r in zip(request.texts, results)]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)