import json

from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any


# Load default settings for summary types
summary_defaults = json.load(open(Path(__file__).parent.parent / "configs/defaults.json"))

# Define supported default summary types found in defaults.json
class SummaryType(str, Enum):
    description = "description"
    outline = "outline"
    slug = "slug"
    subtitle = "subtitle"
    title = "title"

class SummaryResponse(BaseModel):
    success: bool = Field(..., description="The success status of the summary operation")
    summary: list[str] = Field(..., description="The generated summary as a list of strings, one for each key point")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata related to the summary")
    warnings: list[str] = Field(default_factory=list, description="A List of warnings encountered during summary generation")

class SummaryRequest(BaseModel):
    content: str = Field(..., description="The markdown text content to be summarized")
    summary_type: SummaryType = Field(None, description="The type of summary to generate determines supplied arguments")
    max_length: Optional[int] = Field(None, description="The maximum length of the summary in tokens")
    min_length: Optional[int] = Field(None, description="The minimum length of the summary in tokens")
    max_sentences: Optional[int] = Field(None, description="The maximum number of sentences in the summary")
    prompt: Optional[str] = Field("Provide a detailed summary", description="The prompt to guide the summary generation")
    tone: Optional[str] = Field("professional", description="The tone of the summary, e.g., professional, casual, etc.")
    temperature: Optional[float] = Field(0.1, description="The temperature setting for the language model to control creativity")
    page_uri: Optional[str] = Field(None, description="The URI of the page from which the content was extracted")
    store_results: bool = Field(True, description="Whether to store the summary results in the database")
    
    @model_validator(mode='after')
    def apply_type_defaults(self) -> dict:
        """Apply default values based on the specified summary type"""
        model_dict = self.model_dump(exclude_none=True)

        # If a summary type is specified, overwrite defaults with user supplied values
        if self.summary_type and self.summary_type.value in SummaryType:
            default_dict = summary_defaults.get(self.summary_type.value, {})
            # Remove summary_type if it exists to prevent recursion
            default_dict.pop("summary_type", None)
            # Validate defaults against the model
            default_model = SummaryRequest(**default_dict)
            # Apply user specified values over defaults
            model_dict = default_model.model_dump()
            model_dict.update(model_dict)

        return model_dict
