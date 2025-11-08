import types
import numpy
import pytest
import spacy

from app.core.utils import models


@pytest.mark.parametrize("getter, args, expected_type", [
    (models.get_acceptability, ["test"], list),
    (models.get_classifier, ["test"], dict),
    (models.get_embedding, ["test"], numpy.ndarray),
    (models.get_generator, ["test"], str),
    (models.get_keybert, ["test"], list),
    (models.get_polarity, ["test"], dict),
    (models.get_sentiment, ["test"], dict),
    (models.get_spam, ["test"], list),
    (models.get_toxicity, ["test"], list),
    (models.get_yake, ["test"], list),
])
def test_models(monkeypatch, getter, args, expected_type):
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
    real_func = getter()
    assert callable(mock_func)

    real_result = real_func("test")
    assert isinstance(real_result, expected_type)

    # Restore original debug setting
    monkeypatch.setattr(models.settings, "debug", original_debug)
