from fastapi import APIRouter
from backend.models.schemas import QueryRequest, QueryResponse, Source
from backend.services.rag import answer_question

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    answer, sources, stats = answer_question(req.question, model=req.model, provider=req.provider)
    return QueryResponse(
        answer=answer,
        sources=[Source(**s) for s in sources],
        retrieval_time=stats["retrieval_time"],
        llm_time=stats["llm_time"]
    )
