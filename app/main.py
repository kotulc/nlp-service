import spacy
import torch
import transformers

from contextlib import asynccontextmanager

from fastapi import FastAPI

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.config import get_settings
from app.database import init_database
from app.routes import metrics, summary, tags


# Extract constants from settings
settings = get_settings()
DEFAULT_MODEL = settings.defaults.models.default
DEFAULT_TEMPLATE = settings.defaults.template
DEFAULT_KWARGS = settings.defaults.transformers.model_dump()

# Define shared application resources
resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database connection
    init_database()

    # Initialize the content generation model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    resources['generator'] = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Initialize all remaining resources here
    resources['embedding_model'] = SentenceTransformer('all-MiniLM-L6-v2')
    resources['key_bert'] = KeyBERT('all-MiniLM-L6-v2')
    resources['spacy_nlp'] = spacy.load("en_core_web_lg")
    resources['toxic_classifier'] = pipeline("text-classification", model="unitary/toxic-bert")
    resources['zero_shot_pipe'] = pipeline(model='facebook/bart-large-mnli')
    yield

    # Clean up resources here
    resources.clear()


# Initialize FastAPI app
settings = get_settings()
app = FastAPI(lifespan=lifespan, title=settings.name, version=settings.version, debug=settings.debug)

# Include all active routers
for endpoints in [metrics, summary, tags]:
    app.include_router(endpoints.router)

