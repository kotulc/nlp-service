"""Unit core provider overrides for fast deterministic execution."""

import pytest

from mdaug.providers import factory as provider_factory
from mdaug.providers.registry import ProviderRegistry


@pytest.fixture(autouse=True)
def use_mock_default_providers(
    request,
    monkeypatch,
    mock_default_registry: ProviderRegistry,
    ):
    """Force unit core tests to resolve defaults to mocks, except registry wiring tests."""
    if request.node.module.__name__.endswith("test_providers"):
        yield
        return

    monkeypatch.setattr(provider_factory, "build_default_registry", lambda: mock_default_registry)
    provider_factory.get_provider_bundle.cache_clear()
    yield
    provider_factory.get_provider_bundle.cache_clear()
