from fastapi import APIRouter

from app.core.tags.tags import TagType, get_tags
from app.schemas.schemas import BaseResponse, get_response
from app.schemas.tags import TagRequest


# Define supported tag argument keys
TAG_ARGS = ('min_length', 'max_length', 'top_n')

# Define the router for tag-related endpoints
tag_names = [tag_type.name for tag_type in TagType]
router = APIRouter(prefix="/tags", tags=tag_names)


@router.post("/", response_model=BaseResponse)
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
