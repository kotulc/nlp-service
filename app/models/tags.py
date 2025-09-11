from enum import Enum
from pydantic import Field
from typing import List

from app.models.schemas import BaseResponse, BaseRequest
from app.core.tags.extract import extract_entities, extract_keywords, extract_related, get_tags


class TagType(str, Enum):
    tags = get_tags             # Extract and return all tag types
    entities = extract_entities # Extract and return only entities
    keywords = extract_keywords # Extract and return keywords only
    related = extract_related   # Extract and return related concepts only


class TagResults(BaseModel):
    tags: Dict[str, List[str]] = Field(..., description="The tags extracted from the supplied content")
    scores: Dict[str, List[float]] = Field(..., description="The scores for each returned tag")


class TagResponse(BaseResponse):
    results: TagResults = Field(..., description="The extracted tags of each requested type")


class TagRequest(BaseRequest):
    tag: TagType = Field(default=TagType.tags, description="Return only tags of the specified types")
    max_length: int | None = Field(None, description="Return related tags with max_length or fewer words")
    min_length: int | None = Field(None, description="Return related tags with min_length or greater words")
    top_n: int | None = Field(None, description="Return the top N tags of each type")
