import types
import numpy
import pytest
import spacy

from app.core.utils import models


@pytest.mark.parametrize("getter, expected_type", [
    (models.get_acceptability, list),
    (models.get_classifier, dict),
    (models.get_embedding, numpy.ndarray),
    (models.get_generator, str),
    (models.get_keybert, list),
    (models.get_polarity, dict),
    (models.get_sentiment, dict),
    (models.get_spam, list),
    (models.get_toxicity, list),
    (models.get_yake, list),
])
def test_models(monkeypatch, getter, expected_type):
    # Save original debug setting
    original_debug = models.settings.debug

    # Test with debug True (mock)
    monkeypatch.setattr(models.settings, "debug", True)
    mock_func = getter()
    assert callable(mock_func)

    mock_result = mock_func("test")
    assert isinstance(mock_result, expected_type)

    # Test with debug False (real)
    monkeypatch.setattr(models.settings, "debug", False)
    getter.cache_clear()
    real_func = getter()
    assert callable(real_func)

    real_result = real_func("test")
    assert isinstance(real_result, expected_type)

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
