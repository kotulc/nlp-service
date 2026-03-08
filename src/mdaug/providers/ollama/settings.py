"""Typed settings models for Ollama generative provider configuration."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mdaug.providers.settings import load_provider_settings_data


DEFAULT_OLLAMA_PROMPTS = {
    "outline": [
        "In as few words as possible, outline the following text",
        "List concise bullet-like points describing the key ideas of the following text",
    ],
    "summarize": [
        "Provide a brief summary of the following text",
        "List several concise 2-8 word summaries for the following text",
    ],
    "tag": [
        "With as few words as possible, list high-level topics from the following text",
        "With as few words as possible, list related concepts from the following text",
    ],
    "title": [
        "In 5 words or less, list concise and engaging titles for the following text",
        "List short headline-style alternatives for the following text",
    ],
}


class OllamaSettingsModel(BaseModel):
    """Base settings model with unknown-field tolerance for forward compatibility."""
    model_config = ConfigDict(extra="ignore")


class OllamaDelimiterSettings(OllamaSettingsModel):
    """Delimiter strings used by operation prompts."""
    outline: str = "<|outline|>:"
    tag: str = "<|tag|>:"
    default: str = "Output:"

    def for_operation(self, operation: str) -> str:
        """Return configured delimiter for an operation, defaulting when absent."""
        value = getattr(self, operation, None)
        if isinstance(value, str) and value:
            return value

        return self.default


class OllamaParsePatternSettings(OllamaSettingsModel):
    """Regex patterns used for response candidate parsing."""
    default: str = r"[.,:;<>\[\]`|\n]"
    outline_tag: str = r"[.,:;<>\[\]`|\n*-]|\*\*|--|---"


class OllamaWordLimitSettings(OllamaSettingsModel):
    """Word-limit constraints for generated candidates."""
    tag_min: int = 1
    tag_max: int = 3
    title_max: int = 8
    summarize_max: int = 18


class OllamaGenerativeProviderSettings(OllamaSettingsModel):
    """Top-level settings for Ollama-backed generation behavior."""
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.2"
    generate_path: str = "/api/generate"
    timeout_seconds: float = 60.0
    request_options: dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 96,
        }
    )
    default_template: str = "{prompt}:\n\nText: {content}\n\n{delimiter}"
    delimiter: OllamaDelimiterSettings = Field(default_factory=OllamaDelimiterSettings)
    prompts: dict[str, list[str]] = Field(default_factory=lambda: DEFAULT_OLLAMA_PROMPTS.copy())
    candidate_min_chars: int = 2
    result_limit: int = 10
    score_precision: int = 4
    word_limits: OllamaWordLimitSettings = Field(default_factory=OllamaWordLimitSettings)
    parse_patterns: OllamaParsePatternSettings = Field(default_factory=OllamaParsePatternSettings)

    @field_validator("base_url", "model", "generate_path", mode="before")
    @classmethod
    def _non_empty_str(cls, value: Any) -> str:
        """Validate and normalize non-empty string settings."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Value must be a non-empty string.")

        return value.strip()

    def prompt_list(self, operation: str) -> list[str]:
        """Return configured prompts for a requested generation operation."""
        prompts = self.prompts.get(operation, [])
        return [prompt for prompt in prompts if isinstance(prompt, str) and prompt.strip()]


def load_settings(config_path: str | Path | None = None) -> OllamaGenerativeProviderSettings:
    """Load and validate merged runtime settings for Ollama provider behavior."""
    settings = load_provider_settings_data("ollama", config_path=config_path)
    return OllamaGenerativeProviderSettings.model_validate(settings)
