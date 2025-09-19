from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, List

from app.schemas.schemas import BaseRequest, BaseResponse
from app.core.metrics import polarity, sentiment, spam, style


class metricsResults(BaseModel):
    diction: Dict[str, float] = Field(..., description="The diction category scores")
    genre: Dict[str, float] = Field(..., description="The genre category scores")
    mode: Dict[str, float] = Field(..., description="The mode category scores")
    tone: Dict[str, float] = Field(..., description="The tone category scores")
    sentiment: Dict[str, float] = Field(..., description="The sentiment category scores")
    polarity: float = Field(..., description="The content polarity score")
    toxicity: float = Field(..., description="The content toxcity score")
    spam: float = Field(..., description="The content spam score")


class SummaryResponse(BaseResponse):
    results: metricsResults = Field(..., description="The computed metrics of each requested type")


class MetricsRequest(BaseRequest):
    metrics: list[str] | None = Field(None, description="Return the metrics of the supplied types")
