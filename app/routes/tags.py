from enum import Enum
from fastapi import APIRouter

from app.schemas.base import BaseResponse, get_response
from app.models.tags import TagRequest, TagType


# Define supported tag argument keys
TAG_ARGS = ('max_length', 'min_length', 'top_n')

# Define the router for tag-related endpoints
tag_names = [tag_type.name for tag_type in TagType]
router = APIRouter(prefix="/tags", tags=tag_names)


@router.post("/", response_model=BaseResponse)
async def get_tags(request: TagRequest, tag_type: Enum=None):
    """Return a response including the metrics of the specified request types""""
    # Parse tag request
    tag_type = request.tag if request.tag else TagType.all_tags
    model = request.model_dump(exclude_none=True)
    kwargs = {k: v for k, v in model.items() if k in TAG_ARGS}
    
    # Define BaseResponse data
    response = get_response(request.content, tag_type, **kwargs)

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


@router.post("/", response_model=BaseResponse)
async def get_title(request: TagRequest):
    return get_tags(request)


@router.post("/entities", response_model=BaseResponse)
async def get_title(request: TagRequest):
    return get_tags(request, tag_type=TagType.entities)


@router.post("/keywords", response_model=BaseResponse)
async def get_subtitle(request: TagRequest):
    return get_tags(request, tag_type=TagType.keywords)


@router.post("/related", response_model=BaseResponse)
async def get_description(request: TagRequest):
    return get_tags(request, tag_type=TagType.related)
