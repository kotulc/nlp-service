import numpy
import spacy
import torch
import transformers
import yake

from keybert import KeyBERT
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from functools import lru_cache

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from app.config import get_settings


# Extract constants from settings
settings = get_settings()
DEFAULT_MODEL = settings.defaults.models.default
DEFAULT_TEMPLATE = settings.defaults.template
DEFAULT_KWARGS = settings.defaults.transformers.model_dump()


class ModelLoader:
    def __init__(self, model_key, default_callable, debug_callable=None):
        self.model_key = model_key
        self.default_callable = default_callable
        self.debug_callable = debug_callable if default_callable else default_callable
        self.remote_endpoint = getattr(settings.models.endpoints, model_key, None)

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


@lru_cache(maxsize=1)
def get_acceptability():
    """Return the acceptability classifier pipeline or a mock function in debug mode"""
    return ModelLoader(
        model_key="acceptability",
        default_callable=pipeline("text-classification", model="textattack/roberta-base-CoLA"),
        debug_callable=lambda *args, **kwargs: [{'label': 'ACCEPTABLE', 'score': 0.9}]
    )


@lru_cache(maxsize=1)
def get_classifier():
    """Return the zero-shot classification pipeline or a mock function in debug mode"""
    return ModelLoader(
        model_key="classifier",
        default_callable=pipeline(model='facebook/bart-large-mnli')
        debug_callable=lambda *args, **kwargs: {"labels": ["mock"], "scores": [1.0]}
    )


@lru_cache(maxsize=1)
def get_embedding():
    """Return the language embedding model or a mock function in debug mode"""
    return ModelLoader(
        model_key="embedding",
        default_callable=SentenceTransformer('all-MiniLM-L6-v2').encode
        debug_callable=lambda *args, **kwargs: numpy.zeros((1, 384))
    )


@lru_cache(maxsize=1)
def get_generator():
    """Return the text generation pipeline or a mock function in debug mode"""
    # Initialize the content generation model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    return ModelLoader(
        model_key="generator",
        default_callable=transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
        debug_callable=lambda *args, **kwargs: "Mock generated content"
    )


@lru_cache(maxsize=1)
def get_keybert():
    """Return the KeyBERT model or a mock function in debug mode"""
    return ModelLoader(
        model_key="keybert",
        default_callable=KeyBERT('all-MiniLM-L6-v2').extract_keywords
        debug_callable=lambda *args, **kwargs: [("mock keyword", 1.0)]
    )


@lru_cache(maxsize=1)
def get_polarity():
    """Return the TextBlob polarity model or a mock function in debug mode"""
    return ModelLoader(
        model_key="polarity",
        default_callable=TextBlob
        debug_callable=lambda x: {'polarity': 0.0, 'subjectivity': 0.0}
    )


@lru_cache(maxsize=1)
def get_sentiment():
    """Return the vader sentiment model or a mock function in debug mode"""
    return ModelLoader(
        model_key="sentiment",
        default_callable=SentimentIntensityAnalyzer()
        debug_callable=lambda x: {'neg': 0.5, 'neu': 0.5, 'pos': 0.5, 'compound': -0.5}
    )


@lru_cache(maxsize=1)
def get_spacy():
    """Return the spacy NLP model or a blank model in debug mode"""
    return ModelLoader(
        model_key="spacy",
        default_callable=spacy.load("en_core_web_lg")
        debug_callable=lambda x: spacy.blank("en")
    )


@lru_cache(maxsize=1)
def get_spam():
    """Return the spam classifier tokenizer and model or a mock function in debug mode"""
    return ModelLoader(
        model_key="spam",
        default_callable=AutoModelForSequenceClassification.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
        debug_callable=lambda x: [{'label': 'Spam', 'score': 0.9}]
    )


@lru_cache(maxsize=1)
def get_tokenizer():
    """Return the spam classifier tokenizer and model or a mock function in debug mode"""
    return ModelLoader(
        model_key="tokenizer",
        default_callable=tokenizer = AutoTokenizer.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
        debug_callable=lambda x: [{'label': 'Spam', 'score': 0.9}]
    )


@lru_cache(maxsize=1)
def get_toxicity():
    """Return the toxicity classifier pipeline or a mock function in debug mode"""
    return ModelLoader(
        model_key="tokenizer",
        default_callable=pipeline("text-classification", model="unitary/toxic-bert")
        debug_callable=lambda x: [{'label': 'Toxic', 'score': 0.9}]
    )


@lru_cache(maxsize=1)
def get_yake():
    """Return the YAKE keyword extractor or a mock function in debug mode"""
    return ModelLoader(
        model_key="yake",
        default_callable=yake.KeywordExtractor(
            lan="en", 
            n=1, 
            dedupLim=0.9, 
            dedupFunc="seqm", 
            top=20, 
            features=None
        ).extract_keywords
        debug_callable=lambda x: [("mock keyword", 1.0)]
    )
