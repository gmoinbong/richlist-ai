"""Create PostgreSQL schema: pgvector extension + billionaires table."""

from dotenv import load_dotenv

from db import DATABASE_URL, init_db

load_dotenv()


def main() -> None:
    print(f"Connecting to {DATABASE_URL.split('@')[-1]}")
    init_db()
    print("Done: extension 'vector' enabled, table 'billionaires' ready.")


if __name__ == "__main__":
    main()
