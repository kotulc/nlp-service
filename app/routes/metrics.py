from enum import Enum
from fastapi import APIRouter

from app.core.metrics.metrics import MetricsType, get_metrics
from app.schemas.schemas import BaseResponse, get_response
from app.schemas.metrics import MetricsRequest


# Define the router for metrics-related endpoints
metrics_names = [metrics_type.name for metrics_type in MetricsType]
router = APIRouter(prefix="/metrics", tags=metrics_names)


@router.post("/", response_model=BaseResponse)
async def get_metrics(request: MetricsRequest):
    """Return a response including the metrics of the specified request types"""
    # Get a response from the metrics operation
    response = get_response(get_metrics, content=request.content, metrics=request.metrics)

    # Update and store results to the database (optional)
    if request.merge:
        # NOTE: This does not apply to deterministic content
        # Merge existing results from the database
        pass
    if request.commit:
        # Commit new results to the database
        pass

    # Return response
    return response
