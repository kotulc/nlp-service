"""Provider selection and provider-specific settings loading."""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mdaug.service.settings import (
    clear_runtime_settings_cache,
    deep_merge,
    load_runtime_settings,
    load_runtime_settings_with_path,
    read_settings_map,
    resolve_settings_source_path,
)


class ProviderSettingsModel(BaseModel):
    """Base model for provider settings sections."""
    model_config = ConfigDict(extra="ignore")


class ProviderSettings(ProviderSettingsModel):
    """Provider selection names for each provider role."""
    analysis: str = "default"
    extraction: str = "default"
    generative: str = "default"
    relevance: str = "default"

    @field_validator("analysis", "extraction", "generative", "relevance", mode="before")
    @classmethod
    def _provider_name(cls, value: Any) -> str:
        """Validate and normalize provider names."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Provider names must be non-empty strings.")

        return value.strip()


class ProviderConfigEntry(ProviderSettingsModel):
    """Provider-specific configuration source and inline overrides."""
    source: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class ProviderRuntimeConfig(ProviderSettingsModel):
    """Root config sections needed for provider resolution."""
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    provider_configs: dict[str, ProviderConfigEntry] = Field(default_factory=dict)


def clear_provider_settings_cache() -> None:
    """Clear provider and runtime settings caches."""
    clear_runtime_settings_cache()
    get_provider_settings.cache_clear()


def clear_config_cache() -> None:
    """Backward-compatible alias for clearing provider settings cache."""
    clear_provider_settings_cache()


def load_provider_settings(config_path: str | Path | None = None) -> ProviderSettings:
    """Load configured provider names for each runtime role."""
    runtime_settings = load_runtime_settings(config_path=config_path)
    settings = ProviderRuntimeConfig.model_validate(runtime_settings)
    return settings.providers.model_copy(deep=True)


@lru_cache(maxsize=16)
def get_provider_settings(config_path: str | Path | None = None) -> ProviderSettings:
    """Load cached provider settings from root config and env values."""
    return load_provider_settings(config_path=config_path)


def load_provider_settings_data(
    provider_name: str,
    config_path: str | Path | None = None,
    ) -> dict[str, Any]:
    """Load provider-specific settings from source files and inline overrides."""
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ValueError("provider_name must be a non-empty string.")

    runtime_settings, root_config_path = load_runtime_settings_with_path(config_path=config_path)
    settings = ProviderRuntimeConfig.model_validate(runtime_settings)
    provider_config = settings.provider_configs.get(provider_name.strip())
    if provider_config is None:
        return {}

    source_values: dict[str, Any] = {}
    if provider_config.source:
        source_path = resolve_settings_source_path(provider_config.source, root_config_path)
        source_values = read_settings_map(source_path)

    inline_values = deepcopy(provider_config.settings)
    return deep_merge(source_values, inline_values)


def load_default_provider_settings(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load default provider settings from provider_configs.default."""
    return load_provider_settings_data("default", config_path=config_path)
