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


# TODO: Replace all models and add mock option for debug config in settings
@lru_cache(maxsize=1)
def get_acceptability():
    """Return the acceptability classifier pipeline or a mock function in debug mode"""
    if settings.debug:
        return lambda x: [{'label': 'ACCEPTABLE', 'score': 0.9}]
    return pipeline("text-classification", model="textattack/roberta-base-CoLA")


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
def get_keybert():
    """Return the KeyBERT model or a mock function in debug mode"""
    if settings.debug:
        return lambda *args, **kwargs: [("mock keyword", 1.0)]
    from keybert import KeyBERT
    return KeyBERT('all-MiniLM-L6-v2').extract_keywords


@lru_cache(maxsize=1)
def get_polarity():
    """Return the TextBlob polarity model or a mock function in debug mode"""
    if settings.debug:
        return lambda x: {'polarity': 0.0, 'subjectivity': 0.0}
    return TextBlob


@lru_cache(maxsize=1)
def get_sentiment():
    """Return the vader sentiment model or a mock function in debug mode"""
    if settings.debug:
        return lambda x: {'neg': 0.5, 'neu': 0.5, 'pos': 0.5, 'compound': -0.5}
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def get_spacy():
    """Return the spacy NLP model or a blank model in debug mode"""
    if settings.debug:
        return lambda x: spacy.blank("en")
    return spacy.load("en_core_web_lg")


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


@lru_cache(maxsize=1)
def get_yake():
    """Return the YAKE keyword extractor or a mock function in debug mode"""
    if settings.debug:
        return lambda x: [("mock keyword", 1.0)]
    return yake.KeywordExtractor(
        lan="en", 
        n=1, 
        dedupLim=0.9, 
        dedupFunc="seqm", 
        top=20, 
        features=None
    ).extract_keywords
