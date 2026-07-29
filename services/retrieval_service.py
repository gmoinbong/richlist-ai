from __future__ import annotations

import asyncio

from etl.transform import (
    filter_by_country,
    is_poorest_query,
    is_richest_query,
    match_country,
    sort_by_wealth,
)
from models.types import SearchResult
from repositories.vector_repository import VectorRepository
from services.embedding_service import EmbeddingService


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_repository = vector_repository

    async def search(self, query: str, *, top_k: int = 5) -> SearchResult:
        start = asyncio.get_event_loop().time()
        countries = self._vector_repository.get_countries()
        country = match_country(query, countries)

        if country and (is_richest_query(query) or is_poorest_query(query)):
            rows = self._vector_repository.find_by_country(
                country,
                top_k=top_k,
                descending_wealth=is_richest_query(query),
            )
            billionaires = [row.to_dict() for row in rows]
        else:
            total_count = self._vector_repository.count()
            retrieve_k = min(max(top_k * 10, 50), total_count) if total_count else top_k
            query_embedding = await self._embedding_service.embed_query(query)
            rows = self._vector_repository.search_similar(
                query_embedding,
                top_k=retrieve_k,
                country=country,
            )
            billionaires = [billionaire.to_dict() for billionaire, _ in rows]

            if country and not rows:
                billionaires = filter_by_country(billionaires, country)

            if is_richest_query(query):
                billionaires = sort_by_wealth(billionaires)[:top_k]
            elif is_poorest_query(query):
                billionaires = sort_by_wealth(billionaires, descending=False)[:top_k]
            else:
                billionaires = billionaires[:top_k]

        elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000

        return SearchResult(
            billionaires=billionaires,
            query=query,
            total_found=len(billionaires),
            search_time_ms=elapsed_ms,
        )
