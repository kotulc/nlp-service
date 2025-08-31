from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, List


# Define supported metrics types
class MetricsType(str, Enum):
    diction = "diction"     # Vocabulary, formality and complexity of text
    genre = "genre"         # The overarching literary category
    style = "style"         # The writing style or voice
    tone = "tone"           # The expressed subjectivity (from dogmatic to impartial)
    polarity = "polarity"   # 
    sentiment = "sentiment"
    spam = "spam"
    toxicity = "toxicity"

class MetricsResponse(BaseModel):
    success: bool = Field(..., description="The success status of the metrics computation")
    metrics: Dict[str, Any] = Field(..., description="The computed metrics results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata related to the metrics")
    warnings: list[str] = Field(default_factory=list, description="A List of warnings encountered during metrics computation")

class MetricsRequest(BaseModel):
    content: str = Field(..., description="The markdown text content to compute metrics on")
    metric_types: List[MetricsType] = Field(None, description="Specific types of metrics to compute")
    page_uri: str = Field(None, description="The URI of the page from which the content was extracted")
    store_results: bool = Field(True, description="Whether to store the metrics results in the database")
    
    @model_validator(mode='after')
    def return_dict(self) -> dict:
        """Return model as dict excluding None values"""
        return self.model_dump(exclude_none=True)
