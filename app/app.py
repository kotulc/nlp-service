from fastapi import FastAPI
from app.routes import metrics, summary, tags, workflow
from app.database import create_db_and_tables


# Initialize FastAPI app
app = FastAPI(title="NLP Service")

# Include all active routers
for endpoints in [metrics, summary, tags, workflow]:
    app.include_router(endpoints.router)


@app.on_event("startup")
def on_startup():
    create_db_and_tables()
