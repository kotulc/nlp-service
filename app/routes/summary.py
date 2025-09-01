from enum import Enum
from fastapi import APIRouter

from app.models.base import BaseResponse, get_response
from app.models.summary import SummaryRequest, SummaryType, HEADING_TYPES


# Define supported summary argument keys
SUMMARY_ARGS = ('n_sections', 'top_n')

# Define the router for metrics-related endpoints
summary_types = [summary_type.name for summary_type in SummaryType]
router = APIRouter(prefix="/summary", tags=summary_types)


def get_summary(request: SummaryRequest, summary_type: Enum=None) -> dict:
    """Return a response containing the specified summary types""""
    # Parse summary request arguments
    summary = summary_type if summary_type else request.summary
    model = request.model_dump(exclude_none=True)
    kwargs = {k: v for k, v in model.items() if k in SUMMARY_ARGS}
    
    # Add heading type argument as required by get_headings()
    if summary.name in HEADING_TYPES:
        kwargs['heading'] = summary.name
    
    # Define BaseResponse data
    response = get_response(request.content, summary, **kwargs)

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


@router.post("/", response_model=SummaryResponse)
async def get_title(request: SummaryRequest):
    return get_summary(request)


@router.post("/title", response_model=SummaryResponse)
async def get_title(request: SummaryRequest):
    return get_summary(request, summary_type=SummaryType.title)


@router.post("/subtitle", response_model=SummaryResponse)
async def get_subtitle(request: SummaryRequest):
    return get_summary(request, summary_type=SummaryType.subtitle)


@router.post("/description", response_model=SummaryResponse)
async def get_description(request: SummaryRequest):
    return get_summary(request, summary_type=SummaryType.description)


@router.post("/outline", response_model=SummaryResponse)
async def get_outline(request: SummaryRequest):
    return get_summary(request, summary_type=SummaryType.outline)
