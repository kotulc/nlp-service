## Providers Guide

This package supports provider auto-registration by package convention.
Adding a provider now requires only provider package files and runtime config selection.


## Auto-Registration Rules

- Provider packages live under `src/mdaug/providers/<provider_name>/`.
- Provider registry scans all provider packages automatically.
- For each role, class discovery uses convention:
  - `analysis`: `<ProviderName>AnalysisProvider`
  - `extraction`: `<ProviderName>ExtractionProvider`
  - `generative`: `<ProviderName>GenerativeProvider`
  - `relevance`: `<ProviderName>RelevanceProvider`
- Example for `ollama` package:
  - `OllamaGenerativeProvider` registers for `generative`.


## Minimal Provider Package

Create:

- `src/mdaug/providers/<provider_name>/__init__.py`
- `src/mdaug/providers/<provider_name>/config.yaml`
- `src/mdaug/providers/<provider_name>/settings.py`
- `src/mdaug/providers/<provider_name>/provider.py`

No manual edits are required in `factory.py` for registration.


## Config Behavior

- Root `config.yaml` still selects provider names by role in `providers`.
- `load_provider_settings_data("<provider_name>")` resolves settings from:
  1. `provider_configs.<provider_name>.source` when configured
  2. fallback `src/mdaug/providers/<provider_name>/config.yaml`
  3. merged inline overrides from `provider_configs.<provider_name>.settings`


## Ollama Example

Use Ollama only for generation and keep default for the rest:

```yaml
providers:
  analysis: default
  extraction: default
  generative: ollama
  relevance: default

provider_configs:
  ollama:
    settings:
      model: llama3.1:8b
      base_url: http://127.0.0.1:11434
```

If `provider_configs.ollama.source` is omitted, package defaults are loaded from
`src/mdaug/providers/ollama/config.yaml`.
