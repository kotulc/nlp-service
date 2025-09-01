from enum import Enum
from pydantic import Field

from app.models.base import BaseRequest


# Define supported default tag types
class TagType(str, Enum):
    entities = "entities"
    keywords = "keywords"
    related = "related"


class TagRequest(BaseRequest):
    tags: list[TagType] | None = Field(None, description="Return only tags of the specified types")
    max_length: int | None = Field(None, description="Return related tags with max_length or fewer words")
    min_length: int | None = Field(None, description="Return related tags with min_length or greater words")
    top_n: int | None = Field(None, description="Return the top N tags of each type")
