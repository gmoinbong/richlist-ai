from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer

GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_EMBEDDING_DIM = 768
MINILM_MODEL = "all-MiniLM-L6-v2"
MINILM_DIM = 384


class EmbeddingService:
    def __init__(
        self,
        *,
        use_gemini_embeddings: bool | None = None,
        google_api_key: str | None = None,
    ) -> None:
        if use_gemini_embeddings is None:
            use_gemini_embeddings = (
                os.getenv("USE_GEMINI_EMBEDDINGS", "false").lower() == "true"
            )
        self.use_gemini_embeddings = use_gemini_embeddings
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.embedding_dim = GEMINI_EMBEDDING_DIM if use_gemini_embeddings else MINILM_DIM
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._model: SentenceTransformer | None = None

        if not self.use_gemini_embeddings:
            self._model = SentenceTransformer(MINILM_MODEL)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self._embed_batch([text], task_type="RETRIEVAL_QUERY")
        return embeddings[0]

    def embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        if self.use_gemini_embeddings:
            return asyncio.run(self.embed_documents(texts))

        assert self._model is not None
        encoded = self._model.encode(texts, batch_size=64, show_progress_bar=True)
        return encoded.tolist()

    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, func, *args)

    async def _embed_batch(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        if self.use_gemini_embeddings:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY is required when USE_GEMINI_EMBEDDINGS=true")

            embeddings_model = GoogleGenerativeAIEmbeddings(
                model=GEMINI_EMBEDDING_MODEL,
                google_api_key=self.google_api_key,
            )
            return await embeddings_model.aembed_documents(
                texts,
                task_type=task_type,
                output_dimensionality=GEMINI_EMBEDDING_DIM,
            )

        assert self._model is not None
        encoded = await self._run_in_executor(
            lambda: self._model.encode(texts, batch_size=64, show_progress_bar=False)
        )
        return encoded.tolist()

    def close(self) -> None:
        self._executor.shutdown(wait=False)
