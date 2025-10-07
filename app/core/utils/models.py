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


@lru_cache(maxsize=1)
def get_classifier():
    return pipeline(model='facebook/bart-large-mnli')


@lru_cache(maxsize=1)
def get_embedding():
    return SentenceTransformer('all-MiniLM-L6-v2')


@lru_cache(maxsize=1)
def get_generator():
    # Initialize the content generation model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    return transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)


@lru_cache(maxsize=1)
def get_acceptability():
    # Load a model fine-tuned on the CoLA dataset for linguistic acceptability scoring
    return pipeline("text-classification", model="textattack/roberta-base-CoLA")


@lru_cache(maxsize=1)
def get_spam():
    tokenizer = AutoTokenizer.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
    spam_classifier = AutoModelForSequenceClassification.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
    return tokenizer, spam_classifier


@lru_cache(maxsize=1)
def get_toxicity():
    return pipeline("text-classification", model="unitary/toxic-bert")
