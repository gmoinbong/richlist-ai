from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, Index, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:admin@localhost:5432/mydb")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class Billionaire(Base):
    __tablename__ = "billionaires"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rank: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    wealth_b_usd: Mapped[Optional[float]] = mapped_column(Float)
    country: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    industry: Mapped[Optional[str]] = mapped_column(String(256))
    source: Mapped[Optional[str]] = mapped_column(String(512))
    searchable_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(EMBEDDING_DIM))

    __table_args__ = (
        Index(
            "ix_billionaires_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "Rank": self.rank,
            "Name": self.name,
            "wealth_b_usd": self.wealth_b_usd,
            "country": self.country,
            "industry": self.industry,
            "source": self.source,
            "searchable_text": self.searchable_text,
        }


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
