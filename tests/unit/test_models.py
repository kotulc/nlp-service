import types
import numpy
import pytest
import spacy

from pydantic import BaseModel, Field
from typing import Dict, List

from app.core.utils import models


# Define expected return data types for each model
class Acceptability(BaseModel):
    results: List[dict]

class Classifier(BaseModel):
    results: Dict[str, list]

class Generator(BaseModel):
    results: str

class Keybert(BaseModel):
    results: List[tuple]

class Polarity(BaseModel):
    results: Dict[str, float]

class Sentiment(BaseModel):
    results: Dict[str, float]

class Spam(BaseModel):
    results: List[dict]

class Toxicity(BaseModel):
    results: List[dict]

class Yake(BaseModel):
    results: List[tuple]


@pytest.mark.parametrize("getter, data_model", [
    (models.get_acceptability, Acceptability),
    (models.get_embedding, Embedding),
    (models.get_generator, Generator),
    (models.get_keybert, Keybert),
    (models.get_polarity, Polarity),
    (models.get_sentiment, Sentiment),
    (models.get_spam, Spam),
    (models.get_toxicity, Toxicity),
    (models.get_yake, Yake),
])
def test_models(monkeypatch, getter, data_model):
    # Save original debug setting
    original_debug = models.settings.debug

    # Test with debug False (real)
    monkeypatch.setattr(models.settings, "debug", False)
    getter.cache_clear()
    real_func = getter()
    real_result = real_func(["test"])
    data_model(results=real_result)

    # Restore original debug setting
    monkeypatch.setattr(models.settings, "debug", original_debug)


@pytest.mark.parametrize("getter, data_model", [
    (models.get_acceptability, Acceptability),
    (models.get_embedding, Embedding),
    (models.get_generator, Generator),
    (models.get_keybert, Keybert),
    (models.get_polarity, Polarity),
    (models.get_sentiment, Sentiment),
    (models.get_spam, Spam),
    (models.get_toxicity, Toxicity),
    (models.get_yake, Yake),
])
def test_models_debug(monkeypatch, getter, data_model):
    # Save original debug setting
    original_debug = models.settings.debug

    # Test with debug True (mock)
    monkeypatch.setattr(models.settings, "debug", True)
    getter.cache_clear()
    mock_func = getter()
    mock_result = mock_func("test")
    data_model(results=mock_result)

    # Restore original debug setting
    monkeypatch.setattr(models.settings, "debug", original_debug)


def test_classifier(monkeypatch):
    # Save original debug setting
    original_debug = models.settings.debug

    # Test with debug True (mock)
    monkeypatch.setattr(models.settings, "debug", True)
    mock_func = models.get_classifier()
    mock_result = mock_func("test", candidate_labels=["A", "B"])

    # Test with debug False (real)
    monkeypatch.setattr(models.settings, "debug", False)
    models.get_classifier.cache_clear()
    real_func = models.get_spacy()
    real_result = real_func("test", candidate_labels=["A", "B"])
    assert type(mock_result) == type(real_result)

    Classifier(results=real_result)

    # Restore original debug setting
    monkeypatch.setattr(models.settings, "debug", original_debug)


def test_spacy(monkeypatch):
    # Save original debug setting
    original_debug = models.settings.debug

    # Test with debug True (mock)
    monkeypatch.setattr(models.settings, "debug", True)
    mock_func = models.get_spacy()
    mock_result = mock_func("test")

    # Test with debug False (real)
    monkeypatch.setattr(models.settings, "debug", False)
    models.get_spacy.cache_clear()
    real_func = models.get_spacy()
    real_result = real_func("test")
    assert type(mock_result) == type(real_result)

    # Restore original debug setting
    monkeypatch.setattr(models.settings, "debug", original_debug)
