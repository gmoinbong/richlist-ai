# RichList AI

Semantic search over the Forbes Billionaires List (~3,000 entries).  
Ask a question in plain language — the system finds the right billionaire and answers via Gemini.

## Example queries

```
richest tech billionaire in USA
fashion billionaires from France
who is richer Musk or Arnault
automotive billionaires worth over 50 billion
youngest billionaire in energy
```

## How it works

```
"richest fashion billionaire in France"
        │
        ▼
Query Embedding (MiniLM / Gemini)
        │
        ▼
FAISS — finds top-5 similar records (~50ms)
        │
        ▼
LangChain RAG — Gemini generates the answer
        │
        ▼
"Bernard Arnault, $233B, LVMH, France..."
```

## Data

| | |
|---|---|
| Source | [Forbes Billionaire List 2025 (GitHub)](https://github.com/FilesUploader/Forbes-Billionaire-List) |
| Records | ~3,000 |
| Fields | Rank, Name, Wealth ($B), Country, Source of Wealth, Industry |

## Stack

- Python, FastAPI
- Sentence Transformers + FAISS
- LangChain + Google Gemini 2.5 Flash Lite

## Quick start

```bash
git clone <repo>
cd richlist-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GOOGLE_API_KEY
python download_data.py
uvicorn main:app --reload
```

## API

### Health

```bash
curl http://localhost:8080/health
```

### Ask

```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "richest fashion billionaire in France",
    "top_k": 5,
    "use_gpt": true
  }'
```

### Response

```json
{
  "answer": "The richest fashion billionaire in France is Bernard Arnault...",
  "query": "richest fashion billionaire in France",
  "total_found": 5,
  "search_time_ms": 38.2,
  "matches": [
    {
      "Rank": 1,
      "Name": "Bernard Arnault & family",
      "wealth_b_usd": 233,
      "country": "France",
      "industry": "Fashion & Retail",
      "source": "LVMH"
    }
  ],
  "gemini_used": true
}
```

## Env

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | Google AI API key |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Model for answers |
| `USE_GEMINI_EMBEDDINGS` | `false` | Gemini embeddings vs local MiniLM |
| `USE_LANGCHAIN` | `true` | LangChain RAG chain |
| `PORT` | `8080` | API port |

## Project structure

```
richlist-ai/
├── README.md
├── PROMPT.md              # Cursor environment prompt
├── requirements.txt
├── download_data.py
├── search_engine.py
├── rag_chain.py
├── main.py
└── data/
    └── forbes_billionaires_2025.csv
```

## Why this project

Pet project for hands-on practice with:
- embeddings + vector search
- RAG pipeline with LangChain
- comparing local vs Gemini embeddings
- semantic search on real-world data

3k records is enough for solid retrieval quality. Scaling to 1M is out of scope here.
