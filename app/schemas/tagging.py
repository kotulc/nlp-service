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
    success: bool = Field(..., description="The success status of the tagging operation")
    tags: Dict[str, list[str]] = Field(..., description="The generated tags categorized by tag type")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata related to the tagging")
    warnings: list[str] = Field(default_factory=list, description="A List of warnings encountered during tagging")

class TaggingRequest(BaseModel):
    content: str = Field(..., description="The markdown text content to be tagged")
    tag_types: list[TaggingType] = Field(
        default_factory=lambda: [
            TaggingType.categories, 
            TaggingType.related,
            TaggingType.trending
        ],
        description="The list of tag types to generate"
    )
    max_tags: Optional[int] = Field(8, description="The maximum number of tags to generate per tag type")
    use_mmr: Optional[bool] = Field(True, description="Whether to use Maximal Marginal Relevance for tag selection")
    lambda_mmr: Optional[float] = Field(0.6, description="Lambda parameter to balance relevance and diversity")
    page_uri: Optional[str] = Field(None, description="The URI of the page from which the content was extracted")
    store_results: Optional[bool] = Field(True, description="Whether to store the tagging results in the database")

    @model_validator(mode='after')
    def return_dict(self) -> dict:
        """Return model as dict excluding None values"""
        return self.model_dump(exclude_none=True)
