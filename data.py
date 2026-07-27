from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_EMBEDDING_DIM = 768

FORBES_CSV_URL = (
    "https://raw.githubusercontent.com/FilesUploader/Forbes-Billionaire-List/"
    "main/Forbes%20Billionaire%20List%202025.csv"
)

COUNTRY_ALIASES = {
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "uae": "United Arab Emirates",
}

RICHEST_KEYWORDS = ("richest", "wealthiest", "highest net worth", "most money")
POOREST_KEYWORDS = ("poorest", "lowest net worth")


@dataclass
class SearchResult:
    billionaires: list[dict[str, Any]]
    query: str
    total_found: int
    search_time_ms: float


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Wealth (in $1B USD)": "wealth_b_usd",
        "Wealth (in $ billion USD)": "wealth_b_usd",
        "Country of Citizenship": "country",
        "Source of Wealth": "source",
        "Industry": "industry",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def build_searchable_text(row: pd.Series) -> str:
    parts = [
        str(row.get("Name", "")).strip(),
        f"net worth ${row.get('wealth_b_usd', 'unknown')} billion",
        f"country {row.get('country', 'unknown')}",
        f"industry {row.get('industry', 'unknown')}",
        f"source of wealth {row.get('source', 'unknown')}",
    ]
    return ", ".join(p for p in parts if p)


def _wealth(record: dict[str, Any]) -> float:
    value = record.get("wealth_b_usd", record.get("Wealth (in $1B USD)"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _match_country(query: str, countries: set[str]) -> str | None:
    lowered = query.lower()
    for alias, country in COUNTRY_ALIASES.items():
        if alias in lowered:
            return country

    for country in sorted(countries, key=len, reverse=True):
        if country.lower() in lowered:
            return country
    return None


def _is_richest_query(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in RICHEST_KEYWORDS)


def _is_poorest_query(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in POOREST_KEYWORDS)


def _filter_by_country(records: list[dict[str, Any]], country: str) -> list[dict[str, Any]]:
    country_lower = country.lower()
    return [
        record
        for record in records
        if str(record.get("country", "")).lower() == country_lower
    ]


def _sort_by_wealth(
    records: list[dict[str, Any]],
    *,
    descending: bool = True,
) -> list[dict[str, Any]]:
    return sorted(records, key=_wealth, reverse=descending)


class BillionaireSearchEngine:
    def __init__(
        self,
        csv_path: Path,
        use_gemini_embeddings: bool = False,
        google_api_key: str | None = None,
    ):
        self.csv_path = csv_path
        self.use_gemini_embeddings = use_gemini_embeddings
        self.google_api_key = google_api_key
        self.embedding_dim = GEMINI_EMBEDDING_DIM if use_gemini_embeddings else 384
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model: SentenceTransformer | None = None
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.records: list[dict[str, Any]] = []
        self.texts: list[str] = []

    async def initialize(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.csv_path}. Run: python download_data.py"
            )

        df = pd.read_csv(self.csv_path)
        df = normalize_columns(df)
        df = df.dropna(subset=["Name"])
        df["searchable_text"] = df.apply(build_searchable_text, axis=1)

        self.records = df.to_dict(orient="records")
        self.texts = df["searchable_text"].tolist()
        self.countries = {
            str(country)
            for country in df["country"].dropna().unique()
        }

        if not self.use_gemini_embeddings:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = await self._embed_batch(self.texts, task_type="RETRIEVAL_DOCUMENT")
        self.index.add(np.array(embeddings, dtype="float32"))

    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

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

        assert self.model is not None
        encoded = await self._run_in_executor(
            lambda: self.model.encode(texts, batch_size=64, show_progress_bar=False)
        )
        return encoded.tolist()

    async def search(self, query: str, top_k: int = 5) -> SearchResult:
        start = asyncio.get_event_loop().time()
        country = _match_country(query, self.countries)

        if country and (_is_richest_query(query) or _is_poorest_query(query)):
            billionaires = _filter_by_country(self.records, country)
            billionaires = _sort_by_wealth(
                billionaires,
                descending=_is_richest_query(query),
            )[:top_k]
        else:
            retrieve_k = min(max(top_k * 10, 50), len(self.records))
            query_embeddings = await self._embed_batch([query], task_type="RETRIEVAL_QUERY")
            query_vec = np.array([query_embeddings[0]], dtype="float32")
            _, indices = await self._run_in_executor(self.index.search, query_vec, retrieve_k)

            billionaires = [self.records[idx] for idx in indices[0] if idx >= 0]
            if country:
                billionaires = _filter_by_country(billionaires, country)

            if _is_richest_query(query):
                billionaires = _sort_by_wealth(billionaires)[:top_k]
            elif _is_poorest_query(query):
                billionaires = _sort_by_wealth(billionaires, descending=False)[:top_k]
            else:
                billionaires = billionaires[:top_k]

        elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000

        return SearchResult(
            billionaires=billionaires,
            query=query,
            total_found=len(billionaires),
            search_time_ms=elapsed_ms,
        )

    def close(self) -> None:
        self.executor.shutdown(wait=True)
