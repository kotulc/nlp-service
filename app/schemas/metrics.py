from pydantic import BaseModel, Field
from typing import Dict

from app.schemas.schemas import BaseRequest, BaseResponse


class MetricsResult(BaseModel):
    diction: Dict[str, float] | None = Field(default=None, description="The diction category scores")
    genre: Dict[str, float] | None = Field(default=None, description="The genre category scores")
    mode: Dict[str, float] | None = Field(default=None, description="The mode category scores")
    tone: Dict[str, float] | None = Field(default=None, description="The tone category scores")
    sentiment: Dict[str, float] | None = Field(default=None, description="The sentiment category scores")
    polarity: float | None = Field(default=None, description="The content polarity score")
    toxicity: float | None = Field(default=None, description="The content toxcity score")
    spam: float | None = Field(default=None, description="The content spam score")


class MetricsResponse(BaseResponse):
    result: MetricsResult = Field(..., description="The computed metrics of each requested type")


class MetricsRequest(BaseRequest):
    metrics: list[str] | None = Field(None, description="Return the metrics of the supplied types")
