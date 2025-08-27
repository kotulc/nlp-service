from fastapi import FastAPI
from app.routes import metrics, summary, tags, workflow


# Initialize FastAPI app
app = FastAPI(title="NLP Service")

# Include all active routers
for endpoints in [metrics, summary, tags, workflow]:
    app.include_router(endpoints.router)
