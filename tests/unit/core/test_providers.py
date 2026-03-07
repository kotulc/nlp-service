"""Unit tests for provider registry, config, and factory behavior."""

from pathlib import Path

import pytest

from mdaug.providers.settings import (
    ProviderSettings,
    clear_config_cache,
    load_default_provider_settings,
    load_provider_settings_data,
    load_provider_settings,
)
from mdaug.providers.default.settings import load_settings
from mdaug.providers.factory import build_default_registry, create_provider_bundle
from mdaug.providers.registry import ProviderRegistry


def test_provider_registry_register_and_resolve():
    """Registered provider factories resolve by role and provider name."""
    registry = ProviderRegistry()
    registry.register("analysis", "demo", lambda: {"provider": "demo"})

    factory = registry.resolve("analysis", "demo")
    assert factory() == {"provider": "demo"}


def test_provider_registry_unknown_role_or_name():
    """Resolving unknown role or provider name raises KeyError."""
    registry = ProviderRegistry()

    with pytest.raises(KeyError):
        registry.resolve("analysis", "default")

    registry.register("analysis", "default", object)
    with pytest.raises(KeyError):
        registry.resolve("analysis", "missing")


def test_load_provider_settings_from_config_file(tmp_path: Path):
    """Provider settings are loaded from config yaml providers section."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "providers:",
                "  analysis: default",
                "  extraction: default",
                "  generative: default",
                "  relevance: default",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_provider_settings(config_path=config_path)

    assert settings == ProviderSettings(
        analysis="default",
        extraction="default",
        generative="default",
        relevance="default",
    )


def test_load_provider_settings_reads_only_config_values(tmp_path: Path):
    """Provider settings are sourced from config when values are provided."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "providers:",
                "  analysis: custom_analysis",
                "  extraction: default",
                "  generative: default",
                "  relevance: default",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_provider_settings(config_path=config_path)

    assert settings == ProviderSettings(
        analysis="custom_analysis",
        extraction="default",
        generative="default",
        relevance="default",
    )


def test_load_provider_settings_falls_back_to_defaults_when_missing_config(tmp_path: Path):
    """Missing config falls back to built-in default provider names."""
    config_path = tmp_path / "missing.yaml"
    settings = load_provider_settings(config_path=config_path)

    assert settings == ProviderSettings()


def test_load_default_provider_settings_includes_model_and_provider_defaults():
    """Default provider settings expose model and provider module defaults."""
    settings = load_default_provider_settings()

    assert settings["models"]["generative"]["model_name"] == "sshleifer/tiny-gpt2"
    assert settings["models"]["generative"]["kwargs"]["max_new_tokens"] == 96
    assert settings["relevance"]["mmr"]["top_n"] == 10


def test_load_default_provider_settings_reads_config_overrides(tmp_path: Path):
    """Local config.yaml can override nested default provider model settings."""
    provider_config_path = tmp_path / "provider.yaml"
    provider_config_path.write_text(
        "\n".join(
            [
                "models:",
                "  generative:",
                "    model_name: source-model",
                "    kwargs:",
                "      max_new_tokens: 120",
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "provider_configs:",
                "  default:",
                f"    source: {provider_config_path.name}",
                "    settings:",
                "      models:",
                "        generative:",
                "          model_name: custom-model",
                "          kwargs:",
                "            max_new_tokens: 128",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_default_provider_settings(config_path=config_path)

    assert settings["models"]["generative"]["model_name"] == "custom-model"
    assert settings["models"]["generative"]["kwargs"]["max_new_tokens"] == 128


def test_load_default_provider_settings_reads_env_overrides(monkeypatch):
    """Environment overrides merge into nested default provider settings."""
    monkeypatch.setenv(
        "MDAUG_PROVIDER_CONFIGS__DEFAULT__SETTINGS__MODELS__GENERATIVE__KWARGS__MAX_NEW_TOKENS",
        "144",
    )

    clear_config_cache()
    try:
        settings = load_default_provider_settings()
    finally:
        clear_config_cache()

    assert settings["models"]["generative"]["kwargs"]["max_new_tokens"] == 144


def test_default_provider_settings_model_parses_typed_values(monkeypatch):
    """Pydantic default-provider settings expose typed values for provider modules."""
    monkeypatch.setenv(
        "MDAUG_PROVIDER_CONFIGS__DEFAULT__SETTINGS__PROVIDER__ANALYSIS__ROUND_DIGITS",
        "3",
    )
    monkeypatch.setenv(
        "MDAUG_PROVIDER_CONFIGS__DEFAULT__SETTINGS__RELEVANCE__MMR__SIM_LAMBDA",
        "0.7",
    )

    clear_config_cache()
    try:
        settings = load_settings()
    finally:
        clear_config_cache()

    assert settings.provider.analysis.round_digits == 3
    assert settings.relevance.mmr.sim_lambda == 0.7


def test_create_provider_bundle_with_default_registry():
    """Provider bundle creation returns default provider selection from settings."""
    settings = ProviderSettings()
    bundle = create_provider_bundle(settings=settings, registry=build_default_registry())

    assert bundle.names.analysis == "default"
    assert bundle.names.extraction == "default"
    assert bundle.names.generative == "default"
    assert bundle.names.relevance == "default"


def test_create_provider_bundle_unknown_provider_raises():
    """Unknown configured provider names raise KeyError during bundle creation."""
    settings = ProviderSettings(analysis="missing")

    with pytest.raises(KeyError):
        create_provider_bundle(settings=settings, registry=build_default_registry())


def test_build_default_registry_includes_default_provider_names():
    """Default registry exposes only default provider names for each role."""
    registry = build_default_registry()

    assert registry.resolve("analysis", "default")
    assert registry.resolve("extraction", "default")
    assert registry.resolve("generative", "default")
    assert registry.resolve("relevance", "default")

    with pytest.raises(KeyError):
        registry.resolve("analysis", "mock")


def test_load_provider_settings_data_unknown_provider_name():
    """Unknown provider names resolve to an empty settings mapping."""
    settings = load_provider_settings_data("missing")
    assert settings == {}
