from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_rag_service
from api.schemas import QueryRequest, QueryResponse
from services.rag_service import RagService

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/")
def root() -> dict[str, str]:
    return {"message": "RichList AI is running"}


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    rag_service: RagService = Depends(get_rag_service),
) -> QueryResponse:
    answer, results = await rag_service.generate_answer(body.question, top_k=body.top_k)
    return QueryResponse(
        answer=answer,
        query=results.query,
        total_found=results.total_found,
        search_time_ms=results.search_time_ms,
        matches=results.billionaires,
    )
