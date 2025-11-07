import numpy
import torch
import transformers

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


# TODO: Replace all models and add mock option for debug config in settings
@lru_cache(maxsize=1)
def get_classifier():
    """Return the zero-shot classification pipeline or a mock function in debug mode"""
    if settings.debug:
        return lambda *args, **kwargs: {"labels": ["mock"], "scores": [1.0]}
    return pipeline(model='facebook/bart-large-mnli')


@lru_cache(maxsize=1)
def get_embedding():
    """Return the language embedding model or a mock function in debug mode"""
    if settings.debug:
        return lambda *args, **kwargs: numpy.zeros((1, 384))
    return SentenceTransformer('all-MiniLM-L6-v2')


@lru_cache(maxsize=1)
def get_generator():
    """Return the text generation pipeline or a mock function in debug mode"""
    if settings.debug:
        return lambda x: "Mock generated content"
    # Initialize the content generation model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    return transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)


@lru_cache(maxsize=1)
def get_acceptability():
    """Return the acceptability classifier pipeline or a mock function in debug mode"""
    if settings.debug:
        return lambda x: [{'label': 'ACCEPTABLE', 'score': 0.9}]
    return pipeline("text-classification", model="textattack/roberta-base-CoLA")


@lru_cache(maxsize=1)
def get_spam():
    """Return the spam classifier tokenizer and model or a mock function in debug mode"""
    if settings.debug:
        return lambda x: [{'label': 'Spam', 'score': 0.9}]
    tokenizer = AutoTokenizer.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
    spam_classifier = AutoModelForSequenceClassification.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
    return tokenizer, spam_classifier


@lru_cache(maxsize=1)
def get_toxicity():
    """Return the toxicity classifier pipeline or a mock function in debug mode"""
    if settings.debug:
        return lambda x: [{'label': 'Toxic', 'score': 0.9}]
    return pipeline("text-classification", model="unitary/toxic-bert")
