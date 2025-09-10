from fastapi import FastAPI

from app.config import get_settings
from app.database import init_database
from app.routes import metrics, summary, tags


# Initialize FastAPI app
settings = get_settings()
app = FastAPI(title=settings.name, version=settings.version, debug=settings.debug)

# Include all active routers
for endpoints in [metrics, summary, tags]:
    app.include_router(endpoints.router)


@app.on_event("startup")
def on_startup():
    init_database()
