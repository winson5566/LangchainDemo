from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    """
    Request schema for the /query endpoint.

    Attributes:
        question (str): The user's input question to be answered.
        model (Optional[str]): The name of the LLM model to use (default: "gpt-5-mini").
        provider (str): The LLM provider, such as "openai", "claude", or "local".
    """
    question: str
    model: Optional[str] = "gpt-5-mini"
    provider: str = "openai"


class Source(BaseModel):
    """
    Schema representing metadata for a retrieved document source.

    Attributes:
        source (str): The source document name or identifier.
        page (Optional[int]): Page number where the content was found (if applicable).
        score (Optional[float]): Relevance score (if available from retriever).
        snippet (Optional[str]): Text snippet from the source that was used in context.
    """
    source: str
    page: Optional[int] = None
    score: Optional[float] = None
    snippet: Optional[str] = None


class QueryResponse(BaseModel):
    """
    Response schema for the /query endpoint.

    Attributes:
        answer (str): The generated answer from the LLM.
        sources (List[Source]): List of document sources used to answer the question.
        retrieval_time (Optional[float]): Time spent on document retrieval (in milliseconds).
        llm_time (Optional[float]): Time spent generating the answer with the LLM (in milliseconds).
    """
    answer: str
    sources: List[Source] = []
    retrieval_time: Optional[float] = None
    llm_time: Optional[float] = None
