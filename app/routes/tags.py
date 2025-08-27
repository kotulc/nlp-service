from fastapi import APIRouter
from app.schemas.tags import TaggingResponse, TaggingRequest, TaggingType
from app.core.tags import tag


# Define the router for tags-related endpoints
tags = [tag_type.value for tag_type in TaggingType]
router = APIRouter(prefix="/tags", tags=tags)

@router.post("/", response_model=TaggingResponse)
def get_tags(request: TaggingRequest):
    return tag(request)
