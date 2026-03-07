"""Runtime settings loading and merge utilities."""

from copy import deepcopy
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


def as_settings_map(value: object) -> dict[str, Any]:
    """Normalize parsed yaml content into a dictionary."""
    if not isinstance(value, dict):
        return {}

    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries, favoring override values."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
            continue

        merged[key] = value

    return merged


def env_overrides(prefix: str = "MDAUG_") -> dict[str, Any]:
    """Build nested config overrides from environment variables."""
    overrides: dict[str, Any] = {}
    prefix_length = len(prefix)
    for key, raw_value in os.environ.items():
        if not key.startswith(prefix):
            continue

        parts = [part.strip().lower() for part in key[prefix_length:].split("__") if part.strip()]
        if not parts:
            continue

        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value

        current = overrides
        for part in parts[:-1]:
            if not isinstance(current.get(part), dict):
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return overrides


def normalized_config_path(config_path: str | Path | None) -> str | None:
    """Return a stable string cache key for optional config paths."""
    if config_path is None:
        return None

    return str(Path(config_path))


def read_settings_map(path: Path) -> dict[str, Any]:
    """Read a yaml settings file and normalize to a dictionary."""
    if not path.exists():
        return {}

    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    return as_settings_map(parsed)


def resolve_settings_source_path(source: str, config_path: Path) -> Path:
    """Resolve source paths relative to the root config file."""
    source_path = Path(source)
    if source_path.is_absolute():
        return source_path

    return config_path.parent / source_path


@lru_cache(maxsize=16)
def _load_runtime_settings_bundle(config_key: str | None) -> tuple[dict[str, Any], str]:
    """Load merged runtime settings from file and env overrides."""
    config_path = Path(config_key) if config_key else DEFAULT_CONFIG_PATH
    file_values = read_settings_map(config_path)
    runtime_values = deep_merge(file_values, env_overrides())
    return runtime_values, str(config_path)


def clear_runtime_settings_cache() -> None:
    """Clear cached runtime settings after environment or config updates."""
    _load_runtime_settings_bundle.cache_clear()


def load_runtime_settings(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load merged runtime settings from root config and env values."""
    config_key = normalized_config_path(config_path)
    runtime_values, _ = _load_runtime_settings_bundle(config_key)
    return deepcopy(runtime_values)


def load_runtime_settings_with_path(
    config_path: str | Path | None = None,
    ) -> tuple[dict[str, Any], Path]:
    """Load runtime settings and return the resolved root config path."""
    config_key = normalized_config_path(config_path)
    runtime_values, loaded_from = _load_runtime_settings_bundle(config_key)
    return deepcopy(runtime_values), Path(loaded_from)
