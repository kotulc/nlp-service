import pytest

from app.config import get_settings
from app.core.models.loader import loader
from app.core.models.generative import generativeRequest, generativeResponse, get_generative_model


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



def test_generative_request_response_validation(monkeypatch):
    # minimal valid request
    req = generativeRequest(content="This is a short prompt for generation.")

    # ensure debug path is used (lightweight, deterministic)
    monkeypatch.setattr(settings, "debug", True)

    # clear cached loader so setting takes effect
    if hasattr(get_generative_model, "cache_clear"):
        get_generative_model.cache_clear()

    model_callable = get_generative_model()
    assert callable(model_callable)

    # invoke model with the request content and validate response model accepts the output
    result = model_callable(req.content)
    generativeResponse(results=result)
