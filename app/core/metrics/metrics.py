from enum import Enum
from typing import Dict

from app.core.metrics import polarity, sentiment, spam, style


METRIC_TYPES = {
    "diction": style.score_diction,           # Vocabulary, formality and complexity of text
    "genre": style.score_genre,               # The assessed literary category
    "mode": style.score_mode,                 # The writing style or voice
    "tone": style.score_tone,                 # The expressed subjectivity (from dogmatic to impartial)
    "sentiment": sentiment.score_sentiment,   # The negative, neutral, and positive class scores [0.0, 1.0]
    "polarity": polarity.score_polarity,      # The degree of negative or positive sentiment [-1.0, 1.0]
    "toxicity": spam.score_toxicity,          # The computed toxcicity score [0.0, 1.0] 
    "spam": spam.score_spam,                  # The negative and positive spam class scores [0.0, 1.0]
}


def get_metrics(content: str, metrics: list=None) -> dict:
    """Return a dictionary of the requested metrics for the supplied content"""
    results = {}
    
    # Default to all metrics if none are specified
    metrics = metrics if metrics else list(METRIC_TYPES.keys())

    for metric in metrics:
        if metric in METRIC_TYPES:
            metric_function = METRIC_TYPES[metric]
            if metric in ("diction", "genre", "mode", "tone"):
                # Do not include the resulting label for these metrics
                results[metric] = metric_function(content)[0]
            else:
                results[metric] = metric_function(content)

    return results
