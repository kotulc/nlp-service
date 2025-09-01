import uuid

from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List


# Define base request and response classes
class BaseResponse(BaseModel):
    id: int | None = Field(None, description="The id of the returned record")
    success: bool = Field(True, description="The success status of the operation")
    content: Dict[str, Any] = Field(default_factory=dict, description="The computed or generated results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata related to the operation")
     

class BaseRequest(BaseModel):
    content: str = Field(..., description="The text content for the requested operation")
    source: str | None = Field(None, description="The source from which the content was extracted")

    @model_validator(mode='after')
    def return_dict(self) -> dict:
        """Return the model as dictionary excluding None values"""
        return self.model_dump(exclude_none=True)
