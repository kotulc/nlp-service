import pytest


def mock_classifier(content: str, candidate_labels: list, multi_label: bool=False):
    return {
        'labels': candidate_labels,
        'scores': [1.0 / len(candidate_labels)] * len(candidate_labels)
    }


def mock_generator(*args, **kwargs):
    return "Mock generated content"


def mock_acceptability(content: str):
    return lambda x: [{'label': 'ACCEPTABLE', 'score': 0.9}]


def mock_toxicity(content: str):
    return [{'label': 'TOXIC', 'score': 0.9}]


@pytest.fixture(autouse=True)
def mock_models_module(monkeypatch):
    # Patch resource intensive module functions
    monkeypatch.setattr("app.core.utils.models.get_classifier", lambda: mock_classifier)
    monkeypatch.setattr("app.core.utils.models.get_generator", lambda: mock_generator)
    monkeypatch.setattr("app.core.utils.models.get_acceptability", lambda: mock_acceptability)
    monkeypatch.setattr("app.core.utils.models.get_toxicity", lambda: mock_toxicity)

    # Cleanup occurs automatically after test session
    yield
