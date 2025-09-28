from fastapi import APIRouter

from app.core.tags.tags import TAG_TYPES, get_tags
from app.schemas.schemas import get_response
from app.schemas.tags import TagRequest, TagResponse


# Define supported tag argument keys
TAG_ARGS = ('min_length', 'max_length', 'top_n')

# Define the router for tag-related endpoints
tag_names = [tag_type for tag_type in TAG_TYPES.keys()]
router = APIRouter(prefix="/tags", tags=tag_names)


@router.post("/", response_model=TagResponse)
async def post_tags(request: TagRequest):
    """Return a response including the metrics of the specified request types"""
    # Get a response from the tag operation
    response = get_response(
        get_tags, 
        content=request.content, 
        min_length=request.min_length,
        max_length=request.max_length, 
        top_n=request.top_n
    )

    # Update and store results to the database
    pass

    # Return response
    return response
