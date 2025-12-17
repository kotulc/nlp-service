import torch

from pydantic import BaseModel, Field
from typing import Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from functools import lru_cache

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.core.models.loader import ModelLoader


# Define the generative input and output pydantic data models (for testing and validation only)
class sentimentRequest(BaseModel):
    content: str = Field(..., description="The text context used for sentiment analysis")


class sentimentResponse(BaseModel):
    results: Dict[str, float] = Field(None, description="The sentiment label and score")


@lru_cache(maxsize=1)
def get_acceptability_model():
    """Return the acceptability classifier pipeline or a mock function in debug mode"""
    pipe = pipeline("text-classification", model="textattack/roberta-base-CoLA")

    def score_acceptability(content: str) -> float:
        """Compute acceptability score for the supplied string"""
        result = pipe(content)
        return {'score': result[0]['score']}
    
    return ModelLoader(
        model_key="acceptability",
        default_callable=score_acceptability,
        debug_callable=lambda *args, **kwargs: {'score': 0.9}
    )


@lru_cache(maxsize=1)
def get_polarity_model():
    """Return the TextBlob polarity model or a mock function in debug mode"""
    # For both sets of scores: -1 most extreme negative, +1 most extreme positive
    analyzer = SentimentIntensityAnalyzer()
    
    def score_polarity(content: str) -> Dict[str, float]:
        """Compute blob and vader polarity for the supplied string"""
        blob_score = TextBlob(content).sentiment.polarity
        vader_score = analyzer.polarity_scores(content)['compound']
        return {'score': (blob_score + vader_score) / 2}

    return ModelLoader(
        model_key="polarity",
        default_callable=score_polarity,
        debug_callable=lambda *args, **kwargs: {'score': 0.9}
    )


@lru_cache(maxsize=1)
def get_sentiment_model():
    """Return the vader sentiment model or a mock function in debug mode"""
    analyzer = SentimentIntensityAnalyzer()

    def score_sentiment(content: str) -> Dict[str, float]:
        """Compute bart and vader sentiment scores for the supplied string"""
        # TODO: Add BART sentiment model and combine here
        sentiment_scores = analyzer.polarity_scores(content)
        sentiment_scores = {k: sentiment_scores[k] for k in ('neg', 'neu', 'pos')}
        return sentiment_scores
    
    return ModelLoader(
        model_key="sentiment",
        default_callable=score_sentiment,
        debug_callable=lambda *args, **kwargs: {'neg': 0.5, 'neu': 0.5, 'pos': 0.5, 'compound': -0.5}
    )


@lru_cache(maxsize=1)
def get_spam_model():
    """Return the spam classifier tokenizer and model or a mock function in debug mode"""
    spam_tokenizer = AutoTokenizer.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
    spam_classifier = AutoModelForSequenceClassification.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
    
    def score_spam(content: str) -> float:
        """Compute spam scores for the supplied text content"""
        # Tokenize the input
        inputs = spam_tokenizer(content, return_tensors="pt")

        # Get model predictions
        with torch.no_grad():
            outputs = spam_classifier(**inputs)
            logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=1)
        return {'score': probabilities.flatten()[1].item()}

    return ModelLoader(
        model_key="spam",
        default_callable=score_spam,
        debug_callable=lambda *args, **kwargs: {'score': 0.9}
    )


@lru_cache(maxsize=1)
def get_toxicity_model():
    """Return the toxicity classifier pipeline or a mock function in debug mode"""
    pipe = pipeline("text-classification", model="unitary/toxic-bert")

    def score_toxicity(content: str) -> float:
        """Compute toxicity score for the supplied string"""
        result = pipe(content)
        return {'score': result[0]['score']}

    return ModelLoader(
        model_key="tokenizer",
        default_callable=score_toxicity,
        debug_callable=lambda *args, **kwargs: {'score': 0.9}
    )
