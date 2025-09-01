from enum import Enum
from fastapi import APIRouter

from app.models.base import BaseResponse, get_response
from app.models.metrics import MetricsRequest, MetricsType


# Define the router for metrics-related endpoints
metrics_names = [metrics_type.name for metrics_type in MetricsType]
router = APIRouter(prefix="/metrics", tags=metrics_names)


@router.post("/", response_model=BaseResponse)
async def get_metrics(request: MetricsRequest):
    """Return a response including the metrics of the specified request types""""
    # Parse Metrics request
    content = request.content
    metrics = request.metrics if request.metrics else MetricsType

    # Define BaseResponse data
    response = get_response(content, metrics)

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
