from enum import Enum
from pydantic import Field

from app.models.base import BaseRequest
from app.core.tags.extract import extract_entities, extract_keywords, extract_related, get_tags


# Define supported default tag types
class TagType(str, Enum):
    all_tags = get_tags         # Extract and return all tag types
    entities = extract_entities # Extract and return only entities
    keywords = extract_keywords # Extract and return keywords only
    related = extract_related   # Extract and return related concepts only


class TagRequest(BaseRequest):
    tag: TagType = Field(TagType.all_tags, description="Return only tags of the specified types")
    max_length: int | None = Field(None, description="Return related tags with max_length or fewer words")
    min_length: int | None = Field(None, description="Return related tags with min_length or greater words")
    top_n: int | None = Field(None, description="Return the top N tags of each type")
