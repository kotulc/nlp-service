from fastapi import APIRouter
from app.schemas.metrics import MetricsResponse, MetricsRequest, MetricsType
from app.core.metrics import distribution, polarity, sentiment


# Define the router for metrics-related endpoints
tags = [metrics_type.value for metrics_type in MetricsType]
router = APIRouter(prefix="/metrics", tags=tags)

@router.post("/", response_model=MetricsResponse)
async def get_context(request: MetricsRequest):
    return sentiment(request)

@router.post("/context/", response_model=MetricsResponse)
async def get_context(request: MetricsRequest):
    return distribution(request)

@router.post("/granularity/", response_model=MetricsResponse)
async def get_granularity(request: MetricsRequest):
    return granularity(request)

@router.post("/objectivity/", response_model=MetricsResponse)
async def get_objectivity(request: MetricsRequest):
    return objectivity(request)

@router.post("/polarity/", response_model=MetricsResponse)
async def get_polarity(request: MetricsRequest):
    return polarity(request)

@router.post("/sentiment/", response_model=MetricsResponse)
async def get_sentiment(request: MetricsRequest):
    return polarity(request)
