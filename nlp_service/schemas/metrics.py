from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any


# Define supported default metrics types
class MetricsType(str, Enum):
    context = "context"
    granularity = "granularity"
    objectivity = "objectivity"
    polarity = "polarity"
    sentiment = "sentiment"

class MetricsResponse(BaseModel):
    success: bool
    metrics: list[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

class MetricsRequest(BaseModel):
    content: str
    metric_type: MetricsType
