from enum import Enum
from pydantic import Field
from typing import Dict, List

from app.models.schemas import BaseRequest
from app.core.metrics import polarity, sentiment, spam, style


class MetricsType(str, Enum):
    diction = style.score_diction       # Vocabulary, formality and complexity of text
    genre = style.score_genre           # The assessed literary category
    mode = style.score_mode             # The writing style or voice
    tone = style.score_tone             # The expressed subjectivity (from dogmatic to impartial)
    sentiment = sentiment.get_sentiment # The negative, neutral, and positive class scores [0.0, 1.0]
    polarity = polarity.get_polarity    # The degree of negative or positive sentiment [-1.0, 1.0]
    toxicity = spam.score_toxicity      # The computed toxcicity score [0.0, 1.0] 
    spam = spam.score_spam              # The negative and positive spam class scores [0.0, 1.0]


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
    metrics: list[MetricsType] | None = Field(None, description="Return only the metrics of the specified types")
