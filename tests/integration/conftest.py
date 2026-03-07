"""Integration-test provider overrides for fast deterministic execution."""

import pytest

from mdaug.providers import factory as provider_factory
from mdaug.providers.registry import ProviderRegistry


@pytest.fixture(autouse=True)
def use_mock_default_providers(
    monkeypatch,
    mock_default_registry: ProviderRegistry,
    ):
    """Force integration tests to resolve default providers to mock implementations."""
    monkeypatch.setattr(provider_factory, "build_default_registry", lambda: mock_default_registry)
    provider_factory.get_provider_bundle.cache_clear()
    yield
    provider_factory.get_provider_bundle.cache_clear()
