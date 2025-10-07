from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.database import init_database
from app.routes import metrics, summary, tags

from app.config import get_settings
from app.core.utils.models import get_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database connection
    init_database()
    # Load and cache shared application models 
    get_models()
    yield


# Load application settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan, title=settings.name, version=settings.version, debug=settings.debug)

# Include all active routers
for endpoints in [metrics, summary, tags]:
    app.include_router(endpoints.router)
