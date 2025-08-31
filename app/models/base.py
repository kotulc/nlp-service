import uuid

from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List


# Define base content class
class BaseResponse(BaseModel):
    id: uuid.UUID | None = Field(None, description="The UUID of the created or returned record")
    success: bool = Field(..., description="The success status of the operation")
    content: Dict[str, Any] = Field(default_factory=dict, description="The computed or generated results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata related to the operation")
     

class BaseRequest(BaseModel):
    content: str = Field(..., description="The text content for the requested operation")
    items: list[str] | None = Field(None, description="Keys to supply to the requested operation")
    source: str | None = Field(None, description="The source from which the content was extracted")

    @model_validator(mode='after')
    def return_dict(self) -> dict:
        """Return the model as dictionary excluding None values"""
        return self.model_dump(exclude_none=True)
