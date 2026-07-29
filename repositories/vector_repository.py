from __future__ import annotations

from sqlalchemy.orm import Session

from db import Billionaire


class VectorRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def search_similar(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        country: str | None = None,
    ) -> list[tuple[Billionaire, float]]:
        distance = Billionaire.embedding.cosine_distance(query_embedding)
        query = self._db.query(Billionaire, distance.label("distance")).filter(
            Billionaire.embedding.isnot(None)
        )

        if country:
            query = query.filter(Billionaire.country.ilike(country))

        rows = query.order_by(distance).limit(top_k).all()
        return [(billionaire, float(dist)) for billionaire, dist in rows]

    def find_by_country(
        self,
        country: str,
        *,
        top_k: int = 5,
        descending_wealth: bool = True,
    ) -> list[Billionaire]:
        query = self._db.query(Billionaire).filter(Billionaire.country.ilike(country))

        if descending_wealth:
            query = query.order_by(Billionaire.wealth_b_usd.desc().nullslast())
        else:
            query = query.order_by(Billionaire.wealth_b_usd.asc().nullslast())

        return query.limit(top_k).all()

    def get_countries(self) -> set[str]:
        rows = (
            self._db.query(Billionaire.country)
            .filter(Billionaire.country.isnot(None))
            .distinct()
            .all()
        )
        return {str(country) for (country,) in rows if country}

    def bulk_insert(self, records: list[Billionaire]) -> int:
        self._db.add_all(records)
        self._db.commit()
        return len(records)

    def clear(self) -> None:
        self._db.query(Billionaire).delete()
        self._db.commit()

    def count(self) -> int:
        return self._db.query(Billionaire).count()
