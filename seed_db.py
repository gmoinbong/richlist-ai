"""Load Forbes CSV into PostgreSQL with embeddings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from db import Billionaire, SessionLocal, init_db
from etl.transform import build_searchable_text, normalize_columns
from repositories.vector_repository import VectorRepository
from services.embedding_service import EmbeddingService

DATA_PATH = Path(__file__).parent / "data" / "forbes_billionaires_2025.csv"


def seed() -> int:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Run: python download_data.py"
        )

    init_db()

    df = pd.read_csv(DATA_PATH)
    df = normalize_columns(df)
    df = df.dropna(subset=["Name"])
    df["searchable_text"] = df.apply(build_searchable_text, axis=1)

    texts = df["searchable_text"].tolist()
    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed_documents_sync(texts)

    session = SessionLocal()
    try:
        repository = VectorRepository(session)
        repository.clear()

        records = [
            Billionaire(
                rank=_int_or_none(getattr(row, "Rank", None)),
                name=str(getattr(row, "Name", "")).strip(),
                wealth_b_usd=_float_or_none(getattr(row, "wealth_b_usd", None)),
                country=_str_or_none(getattr(row, "country", None)),
                industry=_str_or_none(getattr(row, "industry", None)),
                source=_str_or_none(getattr(row, "source", None)),
                searchable_text=str(getattr(row, "searchable_text", "")),
                embedding=embedding,
            )
            for row, embedding in zip(df.itertuples(index=False), embeddings)
        ]

        repository.bulk_insert(records)
        return repository.count()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        embedding_service.close()


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None and pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None and pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def _str_or_none(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def main() -> None:
    load_dotenv()
    count = seed()
    print(f"Seeded {count} billionaires into PostgreSQL.")


if __name__ == "__main__":
    main()
