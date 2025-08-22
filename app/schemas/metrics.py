from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, List


# Define supported default metrics types
class MetricsType(str, Enum):
    context = "context"
    granularity = "granularity"
    objectivity = "objectivity"
    polarity = "polarity"
    sentiment = "sentiment"

class MetricsResponse(BaseModel):
    success: bool
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

class MetricsRequest(BaseModel):
    content: str
    metric_types: List[MetricsType] = Field(
        default_factory=lambda: [
            MetricsType.context, 
            MetricsType.granularity,
            MetricsType.objectivity,
            MetricsType.polarity,
            MetricsType.sentiment
        ]   
    )
    
    @model_validator(mode='after')
    def return_dict(self) -> dict:
        """Return model as dict excluding None values"""
        return self.model_dump(exclude_none=True)
