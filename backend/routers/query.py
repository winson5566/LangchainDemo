from fastapi import APIRouter
from backend.models.schemas import QueryRequest, QueryResponse, Source
from backend.services.rag import answer_question

# Create a FastAPI router instance for handling RAG-related endpoints
router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Handle POST requests to the /query endpoint.

    This endpoint receives a natural language question and routes it through the
    Retrieval-Augmented Generation (RAG) pipeline. It returns the generated answer,
    supporting document sources, and latency statistics.

    Request Body (QueryRequest):
    - question (str): The user's input question
    - model (str): The LLM model to use (e.g., gpt-5, claude-3)
    - provider (str): The LLM provider (e.g., openai, claude, local)

    Returns:
        QueryResponse:
        - answer (str): Generated response from the LLM
        - sources (List[Source]): List of retrieved document metadata
        - retrieval_time (float): Time spent on document retrieval (ms)
        - llm_time (float): Time spent on LLM inference (ms)
    """
    # Run RAG pipeline with retrieval + generation
    answer, sources, stats = answer_question(
        req.question,
        model=req.model,
        provider=req.provider
    )

    # Format and return structured response
    return QueryResponse(
        answer=answer,
        sources=[Source(**s) for s in sources],
        retrieval_time=stats["retrieval_time"],
        llm_time=stats["llm_time"]
    )
