import json

from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any


# Load default settings for summary types
default_config = json.load(open(Path(__file__).parent / "defaults.json"))
summary_defaults = default_config.get("summary", {})

# Define supported default summary types found in defaults.json
class SummaryType(str, Enum):
    description = "description"
    outline = "outline"
    slug = "slug"
    subtitle = "subtitle"
    title = "title"

class SummaryResponse(BaseModel):
    success: bool
    summary: list[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

class SummaryRequest(BaseModel):
    content: str
    summary_type: SummaryType = None
    max_length: Optional[int] = None
    max_words: Optional[int] = None
    max_sentences: Optional[int] = None
    temperature: Optional[float] = 0.1
    prompt: Optional[str] = "Provide a detailed summary"
    tone: Optional[str] = "professional"

    @model_validator(mode='after')
    def apply_type_defaults(self) -> dict:
        """Apply default values based on the specified summary type"""
        model_dict = self.model_dump(exclude_none=True)

        # If a summary type is specified, overwrite defaults with user specified values
        if self.summary_type and self.summary_type.value in SummaryType:
            default_values = summary_defaults.get(self.summary_type.value, {})
            default_values.update(model_dict)

            return default_values

        return model_dict
