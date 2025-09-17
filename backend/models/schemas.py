from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = "gpt-5-mini"  # ✅ 默认使用 gpt-5-mini
    provider: str = "openai"  # 默认使用 openai

class Source(BaseModel):
    source: str
    page: Optional[int] = None
    score: Optional[float] = None
    snippet: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    retrieval_time: Optional[float] = None
    llm_time: Optional[float] = None
