from fastapi import FastAPI
from app.routers import topics


def create_app() -> FastAPI:
    app = FastAPI(title="NLP Service")
    app.include_router(topics.router)
    return app

app = create_app()
