from fastapi import APIRouter
from ..schemas.summary import SummaryResponse, SummaryRequest, SummaryType
from ..core.summary import summarize


# Define the router for summary-related endpoints
tags = [summary_type.value for summary_type in SummaryType]
router = APIRouter(prefix="/summary", tags=tags)

@router.post("/", response_model=SummaryResponse)
def get_summary(request: SummaryRequest):
    return summarize(request)
