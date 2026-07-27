from dotenv import load_dotenv
from fastapi import FastAPI

from rag_chain import generate_answer

load_dotenv()

app = FastAPI(title="RichList AI", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "RichList AI is running"}


@app.post("/query")
async def query(query: str) -> dict[str, str]:
    return {"message": await generate_answer(query)}
