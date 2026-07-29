from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI

from api.routes import router

load_dotenv()


def create_app() -> FastAPI:
    app = FastAPI(title="RichList AI", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()
