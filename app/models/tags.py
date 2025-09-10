from enum import Enum
from pydantic import Field
from typing import List

from app.models.schemas import BaseResponse, BaseRequest
from app.core.tags.extract import extract_entities, extract_keywords, extract_related, get_tags


class TagType(str, Enum):
    all_tags = get_tags         # Extract and return all tag types
    entities = extract_entities # Extract and return only entities
    keywords = extract_keywords # Extract and return keywords only
    related = extract_related   # Extract and return related concepts only


class TagResults(BaseModel):
    entities: List[str] = Field(..., description="The entities extracted from the supplied content")
    keywords: List[str] = Field(..., description="The keywords extracted from the supplied content")
    keyword_scores: List[float] = Field(..., description="The scores for each returned keyword")
    related: List[str] = Field(..., description="The related concepts extracted from the supplied content") 
    related_scores: List[float] = Field(..., description="The scores for each returned related concept")


class TagResponse(BaseResponse):
    results: TagResults = Field(..., description="The extracted tags of each requested type")


class TagRequest(BaseRequest):
    tag: TagType = Field(TagType.all_tags, description="Return only tags of the specified types")
    max_length: int | None = Field(None, description="Return related tags with max_length or fewer words")
    min_length: int | None = Field(None, description="Return related tags with min_length or greater words")
    top_n: int | None = Field(None, description="Return the top N tags of each type")
