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
DEFAULT_MODEL = settings.models.generative
DEFAULT_TEMPLATE = settings.models.template
DEFAULT_KWARGS = settings.models.transformers.model_dump()


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
# Keyword extraction models
#==================================================================================================

@lru_cache(maxsize=1)
def get_keyword_model(top_n: int=10):
    """Return the keyword extraction model or a mock function in debug mode"""
    key_bert = KeyBERT('all-MiniLM-L6-v2')
    yake_extractor = yake.KeywordExtractor(
            lan="en", 
            n=1, 
            dedupLim=0.9, 
            dedupFunc="seqm", 
            top=top_n // 2, 
            features=None
        )
    
    def extract_keywords(content: str) -> list:
        """Extract bert and yake keywords"""
        bert_keywords = key_bert.extract_keywords(
            content, 
            keyphrase_ngram_range=(1, 1), 
            stop_words='english', 
            top_n=top_n // 2, 
            use_mmr=False
        )
        bert_keywords = [phrase for phrase, score in bert_keywords]

        yake_keywords = yake_extractor.extract_keywords(content)
        yake_keywords = [phrase for phrase, score in yake_keywords]

        # Get unique combined keywords
        keywords = bert_keywords + yake_keywords
        return list({k.lower() for k in keywords})
    
    return ModelLoader(
        model_key="keybert",
        default_callable=extract_keywords,
        debug_callable=lambda *args, **kwargs: [("mock keyword", 1.0)]
    )


#==================================================================================================
# Sentiment analysis models
#==================================================================================================

@lru_cache(maxsize=1)
def get_acceptability_model():
    """Return the acceptability classifier pipeline or a mock function in debug mode"""
    return ModelLoader(
        model_key="acceptability",
        default_callable=pipeline("text-classification", model="textattack/roberta-base-CoLA"),
        debug_callable=lambda *args, **kwargs: [{'label': 'ACCEPTABLE', 'score': 0.9}]
    )


@lru_cache(maxsize=1)
def get_polarity_model():
    """Return the TextBlob polarity model or a mock function in debug mode"""
    return ModelLoader(
        model_key="polarity",
        default_callable=TextBlob
    )


@lru_cache(maxsize=1)
def get_sentiment_model():
    """Return the vader sentiment model or a mock function in debug mode"""
    return ModelLoader(
        model_key="sentiment",
        default_callable=SentimentIntensityAnalyzer.polarity_scores,
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
        return probabilities.flatten()[1]

    return ModelLoader(
        model_key="spam",
        default_callable=score_spam,
        debug_callable=lambda *args, **kwargs: [{'label': 'Spam', 'score': 0.9}]
    )


@lru_cache(maxsize=1)
def get_toxicity_model():
    """Return the toxicity classifier pipeline or a mock function in debug mode"""
    return ModelLoader(
        model_key="tokenizer",
        default_callable=pipeline("text-classification", model="unitary/toxic-bert"),
        debug_callable=lambda *args, **kwargs: [{'label': 'Toxic', 'score': 0.9}]
    )


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
