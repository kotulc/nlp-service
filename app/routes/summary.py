from fastapi import APIRouter
from ..schemas.summary import SummaryResponse, SummaryRequest, SummaryType
from ..core.summary import summarize


# Define the router for summary-related endpoints
tags = [summary_type.value for summary_type in SummaryType]
router = APIRouter(prefix="/summary", tags=tags)

@router.post("/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    return summarize(request)

@router.post("/description/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    request.summary_type = "description"
    return summarize(request)

@router.post("/outline/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    request.summary_type = "outline"
    return summarize(request)

@router.post("/slug/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    request.summary_type = "slug"
    return summarize(request)

@router.post("/subtitle/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    request.summary_type = "subtitle"
    return summarize(request)

@router.post("/title/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    request.summary_type = "title"
    return summarize(request)
