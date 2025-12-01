import types
import numpy
import pytest
import spacy

from pydantic import BaseModel, Field
from typing import Dict, List

from app.core.models import loader


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


@pytest.mark.parametrize("getter", [
    loader.get_acceptability_model,
    loader.get_embedding_model, 
    loader.get_generative_model, 
    loader.get_keybert, 
    loader.get_polarity_model, 
    loader.get_sentiment_model,
    loader.get_document_model,
    loader.get_spam_model,
    loader.get_toxicity_model,
    loader.get_yake,
])
def test_models(monkeypatch, getter):
    # Test with debug False (use real models)
    monkeypatch.setattr(loader.settings, "debug", False)
    getter.cache_clear()
    real_func = getter()
    assert callable(real_func)

    real_result = real_func("test")

    # Test with debug True (use mock models)
    monkeypatch.setattr(loader.settings, "debug", True)
    getter.cache_clear()
    mock_func = getter()
    assert callable(mock_func)

    mock_result = mock_func("test")

    assert real_func is not mock_func
    assert type(real_result) == type(mock_result)


# @pytest.mark.parametrize("getter, data_model", [
#     (models.get_acceptability_model, Acceptability),
#     (models.get_embedding_model, Embedding),
#     (models.get_generative_model, Generator),
#     (models.get_keybert, Keybert),
#     (models.get_polarity_model, Polarity),
#     (models.get_sentiment_model, Sentiment),
#     (models.get_spam_model, Spam),
#     (models.get_toxicity_model, Toxicity),
#     (models.get_yake, Yake),
# ])
# def test_model_results(monkeypatch, getter, data_model):
#     # Save original debug setting
#     original_debug = models.settings.debug

#     # Test with debug True (mock)
#     monkeypatch.setattr(models.settings, "debug", True)
#     getter.cache_clear()
#     mock_func = getter()
#     mock_result = mock_func("test")
#     data_model(results=mock_result)

#     # Restore original debug setting
#     monkeypatch.setattr(models.settings, "debug", original_debug)


# def test_classifier(monkeypatch):
#     # Save original debug setting
#     original_debug = models.settings.debug

#     # Test with debug True (mock)
#     monkeypatch.setattr(models.settings, "debug", True)
#     mock_func = models.get_classifier_model()
#     mock_result = mock_func("test", candidate_labels=["A", "B"])

#     # Test with debug False (real)
#     monkeypatch.setattr(models.settings, "debug", False)
#     models.get_classifier_model.cache_clear()
#     real_func = models.get_document_model()
#     real_result = real_func("test", candidate_labels=["A", "B"])
#     assert type(mock_result) == type(real_result)

#     Classifier(results=real_result)

#     # Restore original debug setting
#     monkeypatch.setattr(models.settings, "debug", original_debug)


# def test_embedding(monkeypatch):
#     # Save original debug setting
#     original_debug = models.settings.debug

#     # Test with debug True (mock)
#     monkeypatch.setattr(models.settings, "debug", True)
#     mock_func = models.get_classifier_model()
#     mock_result = mock_func("test", candidate_labels=["A", "B"])

#     # Test with debug False (real)
#     monkeypatch.setattr(models.settings, "debug", False)
#     models.get_classifier_model.cache_clear()
#     real_func = models.get_document_model()
#     real_result = real_func("test", candidate_labels=["A", "B"])
#     assert type(mock_result) == type(real_result)

#     Classifier(results=real_result)

#     # Restore original debug setting
#     monkeypatch.setattr(models.settings, "debug", original_debug)


# def test_spacy(monkeypatch):
#     # Save original debug setting
#     original_debug = models.settings.debug

#     # Test with debug True (mock)
#     monkeypatch.setattr(models.settings, "debug", True)
#     mock_func = models.get_document_model()
#     mock_result = mock_func("test")

#     # Test with debug False (real)
#     monkeypatch.setattr(models.settings, "debug", False)
#     models.get_document_model.cache_clear()
#     real_func = models.get_document_model()
#     real_result = real_func("test")
#     assert type(mock_result) == type(real_result)

#     # Restore original debug setting
#     monkeypatch.setattr(models.settings, "debug", original_debug)
