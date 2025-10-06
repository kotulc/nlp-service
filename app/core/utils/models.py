import spacy
import torch
import transformers
import yake

from functools import lru_cache

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.config import get_settings


# Extract constants from settings
settings = get_settings()
DEFAULT_MODEL = settings.defaults.models.default
DEFAULT_TEMPLATE = settings.defaults.template
DEFAULT_KWARGS = settings.defaults.transformers.model_dump()


@lru_cache(maxsize=1)
def get_models():
    """Return a dictionary of shared models used across the application"""
    # Define shared model resources
    models = {}

    # Initialize the content generation model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    models['generator'] = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Initialize all remaining models here
    models['embedding_model'] = SentenceTransformer('all-MiniLM-L6-v2')
    models['key_bert'] = KeyBERT('all-MiniLM-L6-v2')
    models['spacy_nlp'] = spacy.load("en_core_web_lg")
    models['spam_classifier'] = AutoModelForSequenceClassification.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
    models['text_blob'] = TextBlob
    models['tokenizer'] = AutoTokenizer.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
    models['toxic_classifier'] = pipeline("text-classification", model="unitary/toxic-bert")
    models['vader_analyzer'] = SentimentIntensityAnalyzer()
    models['zero_shot_pipe'] = pipeline(model='facebook/bart-large-mnli')
    models['yake_extractor'] = yake.KeywordExtractor(
        lan="en", 
        n=1, 
        dedupLim=0.9, 
        dedupFunc="seqm", 
        top=20, 
        features=None
    )

    return models
