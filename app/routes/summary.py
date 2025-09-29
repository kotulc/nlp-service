from enum import Enum
from fastapi import APIRouter

from app.core.summary.summary import SUMMARY_TYPES, get_summary
from app.schemas.schemas import get_response
from app.schemas.summary import SummaryRequest, SummaryResponse


# Define supported summary argument keys
SUMMARY_ARGS = ('n_sections', 'top_n')

# Define the router for metrics-related endpoints
summary_types = list(SUMMARY_TYPES.keys())
router = APIRouter(prefix="/summary", tags=summary_types)


def get_summary_response(request: SummaryRequest) -> dict:
    """Return a response containing the specified summary types"""
    # Get a response from the summary operation
    response = get_response(
        get_summary,
        content=request.content,
        summary=request.summary,
        n_sections=request.n_sections, 
        top_n=request.top_n
    )
    
    # Update and store results to the database
    pass

    # Return response
    return response


@router.post("/", response_model=SummaryResponse)
async def post_summary(request: SummaryRequest):
    return get_summary_response(request)


@router.post("/title", response_model=SummaryResponse)
async def post_title(request: SummaryRequest):
    return get_summary_response(request, summary_type='title')


@router.post("/subtitle", response_model=SummaryResponse)
async def post_subtitle(request: SummaryRequest):
    return get_summary_response(request, summary_type='subtitle')


@router.post("/description", response_model=SummaryResponse)
async def post_description(request: SummaryRequest):
    return get_summary_response(request, summary_type='description')


@router.post("/outline", response_model=SummaryResponse)
async def post_outline(request: SummaryRequest):
    return get_summary_response(request, summary_type='outline')
