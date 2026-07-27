from pathlib import Path

import httpx

from data import FORBES_CSV_URL

DATA_DIR = Path(__file__).parent / "data"
OUTPUT = DATA_DIR / "forbes_billionaires_2025.csv"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading Forbes dataset from GitHub...")
    response = httpx.get(FORBES_CSV_URL, timeout=60)
    response.raise_for_status()
    OUTPUT.write_bytes(response.content)
    print(f"Saved {OUTPUT} ({len(response.content) // 1024} KB)")


if __name__ == "__main__":
    main()
