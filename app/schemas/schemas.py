from pydantic import BaseModel, Field
from typing import Any, Dict


# Define a simple request handling method 
def get_response(operation: callable, **kwargs) -> dict:
    """Supply user content and arguments to each requested operation"""
    # Only pass non-None arguments to the operation
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    # Define BaseResponse return values
    success, result, meta = True, {}, {}

    try:
        # Get all requested enum operations results
        result = operation(**kwargs)
    except Exception as e:
        # Handle all exceptions
        meta[type(e).__name__] = str(e)
        success = False

    return dict(success=success, result=result, metadata=meta)


class BaseResponse(BaseModel):
    id: int | None = Field(None, description="The id of the returned record")
    success: bool = Field(True, description="The success status of the requested operation")
    result: Dict[str, Any] = Field(default_factory=dict, description="The returned results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata related to the operation")
    

class BaseRequest(BaseModel):
    content: str = Field(..., description="The text content for the requested operation")
    source: str | None = Field(None, description="The source from which the content was extracted")
