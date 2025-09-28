from enum import Enum
from fastapi import APIRouter

from app.core.metrics.metrics import METRIC_TYPES, get_metrics
from app.schemas.schemas import get_response
from app.schemas.metrics import MetricsRequest, MetricsResponse


# Define the router for metrics-related endpoints
metrics_names = [metrics_type for metrics_type in METRIC_TYPES.keys()]
router = APIRouter(prefix="/metrics", tags=metrics_names)


@router.post("/", response_model=MetricsResponse)
async def post_metrics(request: MetricsRequest):
    """Return a response including the metrics of the specified request types"""
    # Get a response from the metrics operation
    response = get_response(
        get_metrics, 
        content=request.content, 
        metrics=request.metrics
    )

    # Update and store results to the database
    pass

    # Return response
    return response
