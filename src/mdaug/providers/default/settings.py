"""Typed settings models for default provider configuration."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mdaug.providers.settings import load_provider_settings_data


DEFAULT_GENERATIVE_PROMPTS = {
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


class DefaultSettingsModel(BaseModel):
    """Base settings model that ignores unknown fields for forward compatibility."""
    model_config = ConfigDict(extra="ignore")


class AcceptabilityModelSettings(DefaultSettingsModel):
    """Settings for acceptability scoring model."""
    model_name: str = "textattack/roberta-base-CoLA"


class DocumentModelSettings(DefaultSettingsModel):
    """Settings for document parsing model."""
    model_names: tuple[str, ...] = ("en_core_web_lg", "en_core_web_sm")


class EmbeddingModelSettings(DefaultSettingsModel):
    """Settings for embedding model."""
    model_name: str = "all-MiniLM-L6-v2"


class GenerativeModelSettings(DefaultSettingsModel):
    """Settings for generative model runtime defaults."""
    model_name: str = "sshleifer/tiny-gpt2"
    kwargs: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_new_tokens": 96,
            "num_return_sequences": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }
    )


class KeywordModelSettings(DefaultSettingsModel):
    """Settings for keyword extraction models."""
    embedding_model_name: str = "all-MiniLM-L6-v2"
    default_top_n: int = 10
    keyphrase_ngram_min: int = 1
    keyphrase_ngram_max: int = 1
    stop_words: str = "english"
    use_mmr: bool = False
    yake_language: str = "en"
    yake_ngram_size: int = 1
    yake_dedup_limit: float = 0.9
    yake_dedup_func: str = "seqm"


class PolarityModelSettings(DefaultSettingsModel):
    """Settings for polarity model."""
    pass


class SentimentModelSettings(DefaultSettingsModel):
    """Settings for sentiment model."""
    pass


class SpamModelSettings(DefaultSettingsModel):
    """Settings for spam classification model."""
    model_name: str = "AntiSpamInstitute/spam-detector-bert-MoE-v2.2"
    default_spam_index: int = 1


class ToxicityModelSettings(DefaultSettingsModel):
    """Settings for toxicity classification model."""
    model_name: str = "unitary/toxic-bert"
    top_k: int | None = None


class ModelsSettings(DefaultSettingsModel):
    """Grouped model settings for all default provider components."""
    acceptability: AcceptabilityModelSettings = Field(default_factory=AcceptabilityModelSettings)
    document: DocumentModelSettings = Field(default_factory=DocumentModelSettings)
    embedding: EmbeddingModelSettings = Field(default_factory=EmbeddingModelSettings)
    generative: GenerativeModelSettings = Field(default_factory=GenerativeModelSettings)
    keyword: KeywordModelSettings = Field(default_factory=KeywordModelSettings)
    polarity: PolarityModelSettings = Field(default_factory=PolarityModelSettings)
    sentiment: SentimentModelSettings = Field(default_factory=SentimentModelSettings)
    spam: SpamModelSettings = Field(default_factory=SpamModelSettings)
    toxicity: ToxicityModelSettings = Field(default_factory=ToxicityModelSettings)


class AnalysisProviderSettings(DefaultSettingsModel):
    """Settings for default analysis provider."""
    round_digits: int = 4


class ExtractionProviderSettings(DefaultSettingsModel):
    """Settings for default extraction provider."""
    keyword_top_n: int = 10
    entity_limit: int = 5
    keyword_limit: int = 10


class GenerativeDelimiterSettings(DefaultSettingsModel):
    """Delimiter strings used by default generative provider prompts."""
    outline: str = "<|outline|>:"
    tag: str = "<|tag|>:"
    default: str = "Output:"

    def for_operation(self, operation: str) -> str:
        """Return the delimiter for a supported operation, falling back to default."""
        value = getattr(self, operation, None)
        if isinstance(value, str) and value:
            return value

        return self.default


class GenerativeWordLimitSettings(DefaultSettingsModel):
    """Word-limit constraints for generated candidates."""
    tag_min: int = 1
    tag_max: int = 3
    title_max: int = 8
    summarize_max: int = 18


class GenerativeParsePatternSettings(DefaultSettingsModel):
    """Regex patterns used while parsing generated responses."""
    default: str = r"[.,:;<>\[\]`|\n]"
    outline_tag: str = r"[.,:;<>\[\]`|\n*-]|\*\*|--|---"


class GenerativeProviderSettings(DefaultSettingsModel):
    """Settings for default generative provider behavior."""
    default_template: str = "{prompt}:\n\nText: {content}\n\n{delimiter}"
    delimiter: GenerativeDelimiterSettings = Field(default_factory=GenerativeDelimiterSettings)
    prompts: dict[str, list[str]] = Field(default_factory=lambda: DEFAULT_GENERATIVE_PROMPTS.copy())
    candidate_min_chars: int = 2
    result_limit: int = 10
    word_limits: GenerativeWordLimitSettings = Field(default_factory=GenerativeWordLimitSettings)
    parse_patterns: GenerativeParsePatternSettings = Field(default_factory=GenerativeParsePatternSettings)

    def prompt_list(self, operation: str) -> list[str]:
        """Return configured prompts for an operation."""
        prompts = self.prompts.get(operation, [])
        return [prompt for prompt in prompts if isinstance(prompt, str) and prompt.strip()]


class ProviderSettings(DefaultSettingsModel):
    """Grouped provider-module settings."""
    analysis: AnalysisProviderSettings = Field(default_factory=AnalysisProviderSettings)
    extraction: ExtractionProviderSettings = Field(default_factory=ExtractionProviderSettings)
    generative: GenerativeProviderSettings = Field(default_factory=GenerativeProviderSettings)


class CompositeRelevanceSettings(DefaultSettingsModel):
    """Composite relevance scoring settings."""
    score_precision: int = 4


class SemanticRelevanceSettings(DefaultSettingsModel):
    """Semantic similarity scoring settings."""
    score_precision: int = 4


class MmrRelevanceSettings(DefaultSettingsModel):
    """MMR relevance scoring settings."""
    sim_lambda: float = 0.5
    top_n: int = 10
    score_precision: int = 4


class RelevanceSettings(DefaultSettingsModel):
    """Grouped settings for relevance utilities."""
    composite: CompositeRelevanceSettings = Field(default_factory=CompositeRelevanceSettings)
    semantic: SemanticRelevanceSettings = Field(default_factory=SemanticRelevanceSettings)
    mmr: MmrRelevanceSettings = Field(default_factory=MmrRelevanceSettings)


class DefaultProviderSettings(DefaultSettingsModel):
    """Top-level default provider settings object."""
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    relevance: RelevanceSettings = Field(default_factory=RelevanceSettings)


def load_settings(config_path: str | Path | None = None) -> DefaultProviderSettings:
    """Load and validate default provider settings from merged runtime config."""
    settings = load_provider_settings_data("default", config_path=config_path)
    return DefaultProviderSettings.model_validate(settings)
