from __future__ import annotations

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from models.types import SearchResult
from services.prompt_builder import build_prompt, format_context
from services.retrieval_service import RetrievalService

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


class RagService:
    def __init__(self, retrieval_service: RetrievalService) -> None:
        self._retrieval_service = retrieval_service

    async def generate_answer(self, question: str, *, top_k: int = 5) -> tuple[str, SearchResult]:
        results = await self._retrieval_service.search(question, top_k=top_k)

        if results.total_found == 0:
            return f"No billionaires matched '{question}'. Try another query.", results

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        context = format_context(results)
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            temperature=0.2,
            google_api_key=api_key,
        )

        chain = (
            {
                "question": RunnablePassthrough(),
                "context": lambda _: context,
                "total_found": lambda _: results.total_found,
            }
            | build_prompt()
            | llm
            | StrOutputParser()
        )

        answer = await chain.ainvoke(question)
        return answer, results

    def generate_simple_answer(self, question: str, results: SearchResult) -> str:
        if results.total_found == 0:
            return f"No billionaires matched '{question}'."

        lines = [f"Top matches for '{question}':"]
        for item in results.billionaires[:3]:
            name = item.get("Name", "Unknown")
            wealth = item.get("wealth_b_usd") or item.get("Wealth (in $1B USD)", "?")
            country = item.get("country") or item.get("Country of Citizenship", "?")
            industry = item.get("industry") or item.get("Industry", "?")
            lines.append(f"- {name}: ${wealth}B, {country}, {industry}")

        return "\n".join(lines)
