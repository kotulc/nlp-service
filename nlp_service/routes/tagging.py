from fastapi import APIRouter
from ..schemas.tagging import TaggingResponse, TaggingRequest, TaggingType
from ..core.tagging import tag


# Define the router for tagging-related endpoints
tags = [tag_type.value for tag_type in TaggingType]
router = APIRouter(prefix="/tagging", tags=tags)

@router.post("/", response_model=TaggingResponse)
def get_tagging(request: TaggingRequest):
    return tag(request)
