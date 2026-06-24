# day38_fastapi/models.py
from pydantic import BaseModel
from typing import Optional


class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class AnswerResponse(BaseModel):
    question: str
    answer: str
    session_id: str
    sources_used: int