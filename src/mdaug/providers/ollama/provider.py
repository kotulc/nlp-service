"""Ollama-backed provider implementations for runtime generation behavior."""

import json
import re
from urllib import error, request

from mdaug.providers.interfaces import GenerativeProvider
from mdaug.providers.ollama.settings import OllamaGenerativeProviderSettings, load_settings


def _candidate_filters(
    candidates: list[str],
    settings: OllamaGenerativeProviderSettings,
    operation: str,
    ) -> list[str]:
    """Apply operation-specific candidate word-count filtering."""
    filtered = []
    for candidate in candidates:
        word_count = len(candidate.split())
        in_tag_range = settings.word_limits.tag_min <= word_count <= settings.word_limits.tag_max
        if operation == "tag" and not in_tag_range:
            continue
        if operation == "title" and word_count > settings.word_limits.title_max:
            continue
        if operation == "summarize" and word_count > settings.word_limits.summarize_max:
            continue
        filtered.append(candidate)

    return filtered


def _ollama_generate(prompt: str, settings: OllamaGenerativeProviderSettings) -> str:
    """Send one prompt to Ollama's generate endpoint and return response text."""
    endpoint = f"{settings.base_url.rstrip('/')}{settings.generate_path}"
    payload = {
        "model": settings.model,
        "prompt": prompt,
        "stream": False,
        "options": settings.request_options,
    }
    request_data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=endpoint,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=settings.timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama request failed with status {exc.code}: {detail}") from exc
    except error.URLError as exc:
        reason = str(exc.reason)
        raise RuntimeError(f"Ollama connection failed: {reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Ollama request timed out.") from exc

    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned a non-JSON response.") from exc

    response_text = data.get("response") if isinstance(data, dict) else None
    if not isinstance(response_text, str):
        raise RuntimeError("Ollama response does not include a 'response' string.")

    return response_text


def _parsed_candidates(
    response: str,
    operation: str,
    settings: OllamaGenerativeProviderSettings,
    ) -> list[str]:
    """Parse Ollama response text into cleaned candidate phrases."""
    if operation in {"outline", "tag"}:
        pattern = settings.parse_patterns.outline_tag
    else:
        pattern = settings.parse_patterns.default
    if len(response) < settings.candidate_min_chars:
        return []

    candidates = []
    parts = re.split(pattern, response)
    for part in parts:
        phrase = part.strip()
        if len(phrase) < settings.candidate_min_chars:
            continue
        if not re.search(r"[a-zA-Z]", phrase):
            continue
        candidates.append(" ".join(phrase.split()))

    return list(dict.fromkeys(candidates))


def _rank_candidates(candidates: list[str], score_precision: int) -> dict[str, float]:
    """Assign descending deterministic scores to unique candidates."""
    deduped = list(dict.fromkeys(candidate for candidate in candidates if candidate))
    limited = deduped
    if not limited:
        return {}

    total = len(limited)
    return {
        candidate: round(1.0 - (index / total), score_precision)
        for index, candidate in enumerate(limited)
    }


class OllamaGenerativeProvider(GenerativeProvider):
    """Generative provider using Ollama API calls and configurable parsing rules."""
    def generate(self, content: str, operation: str) -> dict:
        settings = load_settings()
        prompts = settings.prompt_list(operation)
        if not prompts:
            raise ValueError(f"Unsupported generation operation: {operation}")

        delimiter = settings.delimiter.for_operation(operation)

        candidates = []
        for prompt in prompts:
            text_prompt = settings.default_template.format(
                prompt=prompt,
                content=content,
                delimiter=delimiter,
            )
            response = _ollama_generate(text_prompt, settings)
            candidates.extend(_parsed_candidates(response, operation=operation, settings=settings))

        filtered = _candidate_filters(candidates, settings=settings, operation=operation)
        return _rank_candidates(filtered[: settings.result_limit], settings.score_precision)
