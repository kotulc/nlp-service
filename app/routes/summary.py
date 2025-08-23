from fastapi import APIRouter
from ..schemas.summary import SummaryResponse, SummaryRequest, SummaryType
from ..core.summary import generate


# Define the router for summary-related endpoints
tags = [summary_type.value for summary_type in SummaryType]
router = APIRouter(prefix="/summary", tags=tags)

@router.post("/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    return generate(request)

@router.post("/description/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    request.summary_type = "description"
    return generate(request)

@router.post("/list/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    request.summary_type = "list"
    return generate(request)

@router.post("/outline/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    request.summary_type = "outline"
    return generate(request)

@router.post("/slug/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    request.summary_type = "slug"
    return generate(request)

@router.post("/subtitle/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    request.summary_type = "subtitle"
    return generate(request)

@router.post("/title/", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    request.summary_type = "title"
    return generate(request)
