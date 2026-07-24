# day67_fastapi_advanced/models.py
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class TaskType(str, Enum):
    summarize = "summarize"
    classify  = "classify"
    chat      = "chat"


class AIRequest(BaseModel):
    text:      str       = Field(..., min_length=1, max_length=5000)
    task:      TaskType  = TaskType.chat
    options:   Optional[dict] = None


class AIResponse(BaseModel):
    task:        str
    input_text:  str
    result:      str
    tokens_used: Optional[int] = None
    process_time_ms: Optional[float] = None


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=10)
    task:  TaskType  = TaskType.classify