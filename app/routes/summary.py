from enum import Enum
from fastapi import APIRouter

from app.core.summary.summary import SummaryType, get_summary
from app.schemas.schemas import get_response
from app.schemas.summary import SummaryRequest, SummaryResponse


# Define supported summary argument keys
SUMMARY_ARGS = ('n_sections', 'top_n')

# Define the router for metrics-related endpoints
summary_types = [summary_type.name for summary_type in SummaryType]
router = APIRouter(prefix="/summary", tags=summary_types)


def get_summary(request: SummaryRequest) -> dict:
    """Return a response containing the specified summary types"""
    # Get a response from the summary operation
    response = get_response(
        get_summary,
        content=request.content,
        summary=request.summary,
        n_sections=request.n_sections, 
        top_n=request.top_n
    )
    
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
