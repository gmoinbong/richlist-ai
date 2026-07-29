from __future__ import annotations

from typing import Any

import pandas as pd

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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Wealth (in $1B USD)": "wealth_b_usd",
        "Wealth (in $ billion USD)": "wealth_b_usd",
        "Country of Citizenship": "country",
        "Source of Wealth": "source",
        "Industry": "industry",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def build_searchable_text(row: pd.Series) -> str:
    parts = [
        str(row.get("Name", "")).strip(),
        f"net worth ${row.get('wealth_b_usd', 'unknown')} billion",
        f"country {row.get('country', 'unknown')}",
        f"industry {row.get('industry', 'unknown')}",
        f"source of wealth {row.get('source', 'unknown')}",
    ]
    return ", ".join(p for p in parts if p)


def wealth(record: dict[str, Any]) -> float:
    value = record.get("wealth_b_usd", record.get("Wealth (in $1B USD)"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def match_country(query: str, countries: set[str]) -> str | None:
    lowered = query.lower()
    for alias, country in COUNTRY_ALIASES.items():
        if alias in lowered:
            return country

    for country in sorted(countries, key=len, reverse=True):
        if country.lower() in lowered:
            return country
    return None


def is_richest_query(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in RICHEST_KEYWORDS)


def is_poorest_query(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in POOREST_KEYWORDS)


def filter_by_country(records: list[dict[str, Any]], country: str) -> list[dict[str, Any]]:
    country_lower = country.lower()
    return [
        record
        for record in records
        if str(record.get("country", "")).lower() == country_lower
    ]


def sort_by_wealth(
    records: list[dict[str, Any]],
    *,
    descending: bool = True,
) -> list[dict[str, Any]]:
    return sorted(records, key=wealth, reverse=descending)
