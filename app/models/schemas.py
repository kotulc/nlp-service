import uuid
from enum import Enum

from collections.abc import Iterable
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List


# Define a simple request handling method 
def get_response(content: str, operations: list, **kwargs) -> dict:
    """Supply user content and arguments to each requested operation"""
    # Convert supplied operation to iterable 
    if not isinstance(operations, Iterable): operations = [operations]

    # Define BaseResponse data
    success, results, meta = True, {}, {}
    try:
        # Get all requested enum operations results
        for operation in operations:
            assert isinstance(operation, Enum), "Supplied operations must be an enum element"
            results[operation.name] = operation.value(content=content, **kwargs)
    except Exception as e:
        # Handle all exceptions
        meta[str(type(e))] = str(e)
        success = False

    return dict(success=success, results=results, metadata=meta)


class BaseResponse(BaseModel):
    id: int | None = Field(None, description="The id of the returned record")
    success: bool = Field(True, description="The success status of the requested operation")
    results: Dict[str, list] = Field(default_factory=dict, description="The computed or generated results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata related to the operation")
    

class BaseRequest(BaseModel):
    content: str = Field(..., description="The text content for the requested operation")
    source: str | None = Field(None, description="The source from which the content was extracted")
    commit: bool = Field(True, description="Commit returned results to the local database")
    merge: bool = Field(True, description="Merge the results of the operation with those from the local database")
