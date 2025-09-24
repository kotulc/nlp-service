from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, List

from app.schemas.schemas import BaseResponse, BaseRequest


class TagResult(BaseModel):
    tags: Dict[str, List[str]] = Field(..., description="The tags extracted from the supplied content")
    scores: Dict[str, List[float]] = Field(..., description="The scores for each returned tag")


class TagResponse(BaseResponse):
    result: TagResult = Field(..., description="The returned tags and their scores")
    

class TagRequest(BaseRequest):
    max_length: int | None = Field(None, description="Return related tags with max_length or fewer words")
    min_length: int | None = Field(None, description="Return related tags with min_length or greater words")
    top_n: int | None = Field(None, description="Return the top N tags of each type")
