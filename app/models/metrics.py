from enum import Enum
from pydantic import Field

from app.models.base import BaseRequest


# Define supported metrics types
class MetricsType(str, Enum):
    diction = "diction"     # Vocabulary, formality and complexity of text
    genre = "genre"         # The assessed literary category
    mode = "mode"           # The writing style or voice
    tone = "tone"           # The expressed subjectivity (from dogmatic to impartial)
    polarity = "polarity"   # The degree of negative or positive sentiment [-1.0, 1.0]
    sentiment = "sentiment" # The negative, neutral, and positive class scores [0.0, 1.0]
    spam = "spam"           # The negative and positive spam class scores [0.0, 1.0]
    toxicity = "toxicity"   # The computed toxcicity score [0.0, 1.0] 


class MetricsRequest(BaseRequest):
    metrics: list[MetricsType] | None = Field(None, description="Return only the metrics of the specified types")
