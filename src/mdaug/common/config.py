"""Compatibility re-exports for legacy config imports."""

from mdaug.providers.settings import (
    ProviderConfigEntry,
    ProviderRuntimeConfig,
    ProviderSettings,
    clear_config_cache,
    clear_provider_settings_cache,
    get_provider_settings,
    load_default_provider_settings,
    load_provider_settings,
    load_provider_settings_data,
)
from mdaug.service.settings import (
    clear_runtime_settings_cache,
    load_runtime_settings,
    load_runtime_settings_with_path,
)

RuntimeConfig = ProviderRuntimeConfig
load_runtime_config = load_runtime_settings

__all__ = [
    "ProviderConfigEntry",
    "RuntimeConfig",
    "ProviderRuntimeConfig",
    "ProviderSettings",
    "clear_config_cache",
    "clear_provider_settings_cache",
    "clear_runtime_settings_cache",
    "get_provider_settings",
    "load_default_provider_settings",
    "load_provider_settings",
    "load_provider_settings_data",
    "load_runtime_config",
    "load_runtime_settings",
    "load_runtime_settings_with_path",
]
