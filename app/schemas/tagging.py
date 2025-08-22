from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, Optional


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
    tags: Dict[str, list[str]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

class TaggingRequest(BaseModel):
    content: str
    tag_types: list[TaggingType] = Field(
        default_factory=lambda: [
            TaggingType.categories, 
            TaggingType.related,
            TaggingType.trending
        ]   
    )
    use_mmr: Optional[bool] = True
    lambda_mmr: Optional[float] = 0.6
    max_tags: Optional[int] = 16
    prompt: Optional[str] = "Extract relvant and related topics"
    tone: Optional[str] = "neutral"
    temperature: Optional[float] = 0.1

    @model_validator(mode='after')
    def return_dict(self) -> dict:
        """Return model as dict excluding None values"""
        return self.model_dump(exclude_none=True)
