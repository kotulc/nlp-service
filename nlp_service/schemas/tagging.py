from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional


# Define supported default tagging types
class TaggingType(str, Enum):
    categories = "categories"
    entities = "entities"
    keywords = "keywords"
    related = "related"
    themes = "themes"
    trending = "trending"

class TaggingResponse(BaseModel):
    success: bool
    categories: list[str]
    entities: list[str]
    keywords: list[str]
    related: list[str]
    themes: list[str]
    trending: list[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

class TaggingRequest(BaseModel):
    content: str
    content_type: Optional[Literal["document", "section"]] = None
    max_categories: Optional[int] = 4
    max_entities: Optional[int] = 4
    max_keywords: Optional[int] = 8
    max_related: Optional[int] = 8
    max_themes: Optional[int] = 3
    max_trending: Optional[int] = 10
    use_mmr: Optional[bool] = True
    lambda_mmr: Optional[float] = 0.6
    prompt: Optional[str] = "Extract relvant and related topics"
    tone: Optional[str] = "neutral"
    temperature: Optional[float] = 0.1
