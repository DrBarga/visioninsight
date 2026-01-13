from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

class AskResponse(BaseModel):
    analysis_id: str
    question: str
    answer: str
    intent: str
    evidence: Dict[str, Any] = {}
    confidence: float = 0.6
