import numpy
import spacy

from functools import lru_cache

from sentence_transformers import SentenceTransformer
from transformers import pipeline

from app.config import get_settings


# Extract constants from settings
settings = get_settings()


class ModelLoader:
    def __init__(self, model_key, default_callable, debug_callable=None):
        self.model_key = model_key
        self.default_callable = default_callable
        self.debug_callable = debug_callable if default_callable else default_callable
        # self.remote_endpoint = getattr(settings.models.endpoints, model_key, None)
        self.remote_endpoint = None

    def __call__(self, *args, **kwargs):
        # If a remote endpoint is set, route the request there
        if self.remote_endpoint:
            return self._call_remote(*args, **kwargs)
        # If debug is enabled, use the debug callable
        if getattr(settings, "debug", False) and self.debug_callable:
            return self.debug_callable(*args, **kwargs)
        # Otherwise, use the default callable
        return self.default_callable(*args, **kwargs)

    def _call_remote(self, *args, **kwargs):
        # Example: send a POST request to the remote endpoint
        import requests
        payload = {"args": args, "kwargs": kwargs}
        response = requests.post(self.remote_endpoint, json=payload)
        response.raise_for_status()
        return response.json()


#==================================================================================================
# Document and utility models
#==================================================================================================

@lru_cache(maxsize=1)
def get_classifier_model():
    """Return the zero-shot classification pipeline or a mock function in debug mode"""
    return ModelLoader(
        model_key="classifier",
        default_callable=pipeline(model='facebook/bart-large-mnli'),
        debug_callable=lambda *args, **kwargs: {"labels": ["mock"], "scores": [1.0]}
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
