from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from models.types import SearchResult

SYSTEM_PROMPT = """You are RichList AI — a semantic search assistant for the Forbes Billionaires list.

Rules:
- Answer ONLY using the retrieved billionaire records provided in context.
- If data is missing, say you don't know — never invent net worth, country, or industry.
- Prefer concise, conversational answers.
- When comparing people, use only numbers from context.
- When the question asks for richest or poorest, pick by the highest or lowest wealth_b_usd value.
- Mention rank, name, country, industry, and source of wealth when relevant.
"""


def format_context(results: SearchResult) -> str:
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


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "User question: {question}\n\n"
            "Retrieved Forbes records ({total_found} matches):\n{context}\n\n"
            "Write a helpful answer.",
        ),
    ])
