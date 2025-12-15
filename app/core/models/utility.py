import numpy
import spacy

from pydantic import BaseModel, Field
from typing import List

from functools import lru_cache

from sentence_transformers import SentenceTransformer
from transformers import pipeline

from app.core.models.loader import ModelLoader


# Define the generative input and output pydantic data models (for testing and validation only)
class classifierRequest(BaseModel):
    content: str = Field(..., description="The text to classify")
    candidate_labels: List[str] = Field(..., description="The candidate labels to classify the text into")
    kwargs: dict = Field(None, description="Additional arguments for the classifier")


class classifierResponse(BaseModel):
    results: List[float] = Field(..., description="The classification scores for each candidate label")


@lru_cache(maxsize=1)
def get_classifier_model():
    """Return the zero-shot classification pipeline or a mock function in debug mode"""
    pipe = pipeline(model='facebook/bart-large-mnli')

    def score_labels(content: str, candidate_labels: List[str], **kwargs) -> list:
        """Return the classification scores for the candidate labels"""
        result = pipe(content, candidate_labels=candidate_labels, **kwargs)
        scores = {label: score for label, score in zip(result['labels'], result['scores'])} 

        # Return scores in the order the labels were provided
        return [scores[k] for k in candidate_labels]
    
    return ModelLoader(
        model_key="classifier",
        default_callable=score_labels,
        debug_callable=lambda *args, **kwargs: [1.0, 0.0]
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    """Return the language embedding model or a mock function in debug mode"""
    return ModelLoader(
        model_key="embedding",
        default_callable=SentenceTransformer('all-MiniLM-L6-v2').encode,
        debug_callable=lambda *args, **kwargs: numpy.zeros((1, 384))
    )


@lru_cache(maxsize=1)
def get_document_model():
    """Return the spacy NLP model or a blank model in debug mode"""
    return ModelLoader(
        model_key="spacy",
        default_callable=spacy.load("en_core_web_lg"),
    )
