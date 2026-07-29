from __future__ import annotations

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from db import get_db
from repositories.vector_repository import VectorRepository
from services.embedding_service import EmbeddingService
from services.rag_service import RagService
from services.retrieval_service import RetrievalService


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def get_retrieval_service(
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> RetrievalService:
    return RetrievalService(embedding_service, VectorRepository(db))


def get_rag_service(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> RagService:
    return RagService(retrieval_service)
