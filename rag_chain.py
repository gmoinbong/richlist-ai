from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from data import BillionaireSearchEngine, SearchResult

DATA_PATH = Path(__file__).parent / "data" / "forbes_billionaires_2025.csv"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
_engine: BillionaireSearchEngine | None = None


async def _get_engine() -> BillionaireSearchEngine:
    global _engine
    if _engine is None:
        use_gemini = os.getenv("USE_GEMINI_EMBEDDINGS", "false").lower() == "true"
        _engine = BillionaireSearchEngine(
            csv_path=DATA_PATH,
            use_gemini_embeddings=use_gemini,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        await _engine.initialize()
    return _engine


SYSTEM_PROMPT = """You are RichList AI — a semantic search assistant for the Forbes Billionaires list.

Rules:
- Answer ONLY using the retrieved billionaire records provided in context.
- If data is missing, say you don't know — never invent net worth, country, or industry.
- Prefer concise, conversational answers.
- When comparing people, use only numbers from context.
- When the question asks for richest or poorest, pick by the highest or lowest wealth_b_usd value.
- Mention rank, name, country, industry, and source of wealth when relevant.
"""


def _format_context(results: SearchResult) -> str:
    payload = []
    for item in results.billionaires:
        payload.append(
            {
                "rank": item.get("Rank"),
                "name": item.get("Name"),
                "wealth_b_usd": item.get("wealth_b_usd") or item.get("Wealth (in $1B USD)"),
                "country": item.get("country") or item.get("Country of Citizenship"),
                "industry": item.get("industry") or item.get("Industry"),
                "source": item.get("source") or item.get("Source of Wealth"),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def generate_answer(question: str) -> str:
    results = await (await _get_engine()).search(question)

    if results.total_found == 0:
        return f"No billionaires matched '{question}'. Try another query."

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set")

    context = _format_context(results)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "User question: {question}\n\n"
            "Retrieved Forbes records ({total_found} matches):\n{context}\n\n"
            "Write a helpful answer.",
        ),
    ])

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
        | prompt
        | llm
        | StrOutputParser()
    )

    return await chain.ainvoke(question)


def generate_simple_answer(question: str, results: SearchResult) -> str:
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
