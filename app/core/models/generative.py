import torch
import transformers

from pydantic import BaseModel, Field
from typing import Any, List

from functools import lru_cache
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.core.models.loader import ModelLoader

from app.config import get_settings


# Extract constants from settings
settings = get_settings()
DEFAULT_MODEL = settings.models.generative
DEFAULT_KWARGS = settings.models.transformers.model_dump()


# Define the generative input and output pydantic data models (for testing and validation)
class generativeRequest(BaseModel):
    content: str = Field(..., description="The text context used for generation")


class generativeResponse(BaseModel):
    results: List[str] = Field(None, description="The generated text content")


@lru_cache(maxsize=1)
def get_generative_model():
    """Return the text generation pipeline or a mock function in debug mode"""
    # Initialize the content generation model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    generator = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
    default_kwargs = DEFAULT_KWARGS.copy()

    def get_model_inference(content: str, **kwargs) -> List[str]:
        """Return generated text from the model"""
        return generator(content, do_sample=True, return_full_text=False, **default_kwargs, **kwargs)

    return ModelLoader(
        model_key="generator",
        default_callable=get_model_inference,
        debug_callable=lambda *args, **kwargs: ["Mock generated content"]
    )
