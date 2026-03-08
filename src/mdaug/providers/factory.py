"""Provider factory and registry construction via provider package discovery."""

import ast
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import re

from mdaug.providers.interfaces import (
    AnalysisProvider,
    ExtractionProvider,
    GenerativeProvider,
    RelevanceProvider,
)
from mdaug.providers.registry import ProviderRegistry
from mdaug.providers.settings import ProviderSettings, load_provider_settings


PROVIDER_ROLES = ("analysis", "extraction", "generative", "relevance")


@dataclass(frozen=True)
class ProviderBundle:
    """Concrete provider instances selected by provider settings."""
    analysis: AnalysisProvider
    extraction: ExtractionProvider
    generative: GenerativeProvider
    relevance: RelevanceProvider
    names: ProviderSettings


def _class_name_prefix(provider_name: str) -> str:
    """Build expected class-name prefix from snake or hyphen provider names."""
    parts = [part for part in re.split(r"[^a-zA-Z0-9]+", provider_name) if part]
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _module_class_names(module_name: str) -> set[str]:
    """Read top-level class names from provider module source when available."""
    spec = find_spec(module_name)
    if spec is None or spec.origin is None:
        return set()

    module_path = Path(spec.origin)
    if not module_path.exists():
        return set()

    source = module_path.read_text(encoding="utf-8-sig")
    parsed = ast.parse(source)

    return {
        node.name
        for node in parsed.body
        if isinstance(node, ast.ClassDef)
    }


def _provider_class_name(provider_name: str, role: str) -> str:
    """Return convention-based class name for provider role implementations."""
    return f"{_class_name_prefix(provider_name)}{role.capitalize()}Provider"


def _provider_module_name(provider_name: str) -> str:
    """Return import path for a provider package role implementation module."""
    return f"mdaug.providers.{provider_name}.provider"


def _provider_package_names() -> list[str]:
    """Discover provider package names under mdaug.providers."""
    providers_path = Path(__file__).resolve().parent

    names = []
    for child in providers_path.iterdir():
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        if (child / "__init__.py").exists():
            names.append(child.name)

    return sorted(names)


def _resolved_provider_factory(module_name: str, class_name: str):
    """Build lazy factory that resolves provider classes when instantiated."""
    def factory():
        module = import_module(module_name)
        provider_class = getattr(module, class_name, None)
        if provider_class is None:
            raise KeyError(f"Provider class is not defined: {module_name}.{class_name}")

        return provider_class()

    return factory


def build_default_registry() -> ProviderRegistry:
    """Build provider registry by scanning package providers and role conventions."""
    registry = ProviderRegistry()

    for provider_name in _provider_package_names():
        module_name = _provider_module_name(provider_name)
        class_names = _module_class_names(module_name)
        for role in PROVIDER_ROLES:
            class_name = _provider_class_name(provider_name, role)
            if class_name not in class_names:
                continue

            factory = _resolved_provider_factory(module_name, class_name)
            registry.register(role, provider_name, factory)

    return registry


def create_provider_bundle(
    settings: ProviderSettings,
    registry: ProviderRegistry | None = None,
    ) -> ProviderBundle:
    """Create a provider bundle from settings and a provider registry."""
    provider_registry = registry if registry is not None else build_default_registry()

    analysis_factory = provider_registry.resolve("analysis", settings.analysis)
    extraction_factory = provider_registry.resolve("extraction", settings.extraction)
    generative_factory = provider_registry.resolve("generative", settings.generative)
    relevance_factory = provider_registry.resolve("relevance", settings.relevance)

    return ProviderBundle(
        analysis=analysis_factory(),
        extraction=extraction_factory(),
        generative=generative_factory(),
        relevance=relevance_factory(),
        names=settings,
    )


def build_provider_bundle(
    config_path: str | None = None,
    registry: ProviderRegistry | None = None,
    ) -> ProviderBundle:
    """Build a provider bundle using config.yaml provider settings."""
    settings = load_provider_settings(config_path=config_path)
    return create_provider_bundle(settings=settings, registry=registry)


@lru_cache(maxsize=1)
def get_provider_bundle() -> ProviderBundle:
    """Load provider settings from config and build a cached provider bundle."""
    return build_provider_bundle()
