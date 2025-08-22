from fastapi import FastAPI
from .routes import metrics, summary, tagging, workflow


# Initialize FastAPI app
app = FastAPI(title="NLP Service")

# Include all active routers
for endpoints in [metrics, summary, tagging, workflow]:
    app.include_router(endpoints.router)
