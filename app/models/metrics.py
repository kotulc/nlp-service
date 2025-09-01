from enum import Enum
from pydantic import Field

from app.models.base import BaseRequest
from app.core.metrics import polarity, sentiment, spam, style


# Define supported metrics types
class MetricsType(str, Enum):
    diction = style.score_diction       # Vocabulary, formality and complexity of text
    genre = style.score_genre           # The assessed literary category
    mode = style.score_mode             # The writing style or voice
    tone = style.score_tone             # The expressed subjectivity (from dogmatic to impartial)
    polarity = polarity.get_polarity    # The degree of negative or positive sentiment [-1.0, 1.0]
    sentiment = sentiment.get_sentiment # The negative, neutral, and positive class scores [0.0, 1.0]
    toxicity = spam.score_toxicity      # The computed toxcicity score [0.0, 1.0] 
    spam = spam.score_spam              # The negative and positive spam class scores [0.0, 1.0]


class MetricsRequest(BaseRequest):
    metrics: list[MetricsType] | None = Field(None, description="Return only the metrics of the specified types")
