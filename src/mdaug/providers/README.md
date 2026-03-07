## Providers Guide

This guide documents how to add a new provider implementation, using an Ollama-backed
`generative` provider as the example.


## Current Architecture

- Provider contracts are defined in `src/mdaug/providers/interfaces.py`.
- Provider name selection comes from root `config.yaml`:
  - `providers.analysis|extraction|generative|relevance`
- Provider-specific settings come from:
  - `provider_configs.<provider_name>.source`
  - `provider_configs.<provider_name>.settings`
- Provider factories are resolved via `ProviderRegistry` and wired in
  `src/mdaug/providers/factory.py`.


## Target Use Case

Run generation operations (`outline`, `summarize`, `tag`, `title`) against a local Ollama server,
while optionally keeping other roles (`analysis`, `extraction`, `relevance`) on `default`.


## Implementation Steps

1. Create a new provider package:
   - `src/mdaug/providers/ollama/__init__.py`
   - `src/mdaug/providers/ollama/provider.py`
   - `src/mdaug/providers/ollama/settings.py`
   - `src/mdaug/providers/ollama/config.yaml`

2. Define typed provider settings in `ollama/settings.py`:
   - `base_url` (for example `http://127.0.0.1:11434`)
   - `model` (for example `llama3.1:8b`)
   - `timeout_seconds`
   - generation options (`temperature`, `top_p`, `num_predict`, etc.)
   - operation prompt templates and parse/limit settings (match current generative contract)
   - load via `load_provider_settings_data("ollama", config_path=...)` and validate with pydantic.

3. Implement provider class in `ollama/provider.py`:
   - Implement `GenerativeProvider.generate(self, content: str, operation: str) -> dict`.
   - Call Ollama API (typically `POST /api/generate`) with prompt and options.
   - Parse responses into candidate strings.
   - Apply operation-specific filters and limits.
   - Return `{candidate: score}`.
   - Reuse existing ranking for score generation if needed (for example call
     `composite_scores(content, candidates)` from `providers/default/relevance.py`).

4. Register provider in registry wiring:
   - Update `build_default_registry()` in `src/mdaug/providers/factory.py`:
     - `registry.register("generative", "ollama", OllamaGenerativeProvider)`

5. Add provider config entry to root `config.yaml`:

```yaml
providers:
  analysis: default
  extraction: default
  generative: ollama
  relevance: default

provider_configs:
  default:
    source: src/mdaug/providers/default/config.yaml
  ollama:
    source: src/mdaug/providers/ollama/config.yaml
    settings:
      model: llama3.1:8b
      base_url: http://127.0.0.1:11434
```

6. Add tests:
   - Unit tests for settings validation and operation-specific parsing/filtering.
   - Unit tests for Ollama client behavior with mocked HTTP responses.
   - Integration test (optional marker) for real local Ollama.
   - Ensure default test runs still use mocks and do not require Ollama.

7. Document user workflow:
   - Install/start Ollama.
   - Pull model (`ollama pull <model>`).
   - Configure `config.yaml`.
   - Run `mdaug summarize|tag|title|outline`.


## Minimal Code Skeleton

```python
class OllamaGenerativeProvider(GenerativeProvider):
    def generate(self, content: str, operation: str) -> dict:
        settings = load_ollama_settings()
        prompts = settings.prompt_list(operation)
        if not prompts:
            raise ValueError(f"Unsupported generation operation: {operation}")

        candidates: list[str] = []
        for prompt in prompts:
            text = render_prompt(prompt, content, settings)
            response = ollama_generate(text, settings)
            candidates.extend(parse_candidates(response, operation, settings))

        ranked, scores = composite_scores(content, candidates)
        return {candidate: float(score) for candidate, score in zip(ranked, scores)}
```


## What Is Missing Today

- Registry auto-discovery/plugin loading:
  - New providers currently require manual code registration in `factory.py`.
- Role-scoped provider modules:
  - A full provider package can include all roles, but there is no standard helper to register
    role-by-role from a module manifest.
- Provider health checks:
  - No standard preflight command to validate provider connectivity/model availability.
- Provider-specific error taxonomy:
  - Errors currently bubble as generic runtime errors.
- Optional dependency groups:
  - No `extras` group yet for optional provider clients (for example an Ollama python client).
- Provider development template:
  - No scaffold command or template package for quickly adding a new provider.


## Recommended Next Iteration

1. Add a provider manifest pattern (for example `register(registry)` entrypoint per provider
package) and auto-load configured providers.
2. Add `mdaug doctor` or `mdaug providers check` to validate configured providers.
3. Add optional dependency groups in `pyproject.toml` (for example `[project.optional-dependencies]`).
