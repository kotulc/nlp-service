"""Default model-backed provider implementations for command runtime behavior."""

import re

from mdaug.providers.default.models import (
    get_document_model,
    get_generative_model,
    get_keyword_model,
    get_polarity_model,
    get_spam_model,
    get_sentiment_model,
    get_toxicity_model,
)
from mdaug.providers.default.relevance import composite_scores, semantic_similarity
from mdaug.providers.default.settings import GenerativeProviderSettings, load_settings
from mdaug.providers.interfaces import (
    AnalysisProvider,
    ExtractionProvider,
    GenerativeProvider,
    RelevanceProvider,
)


def _candidate_filters(
    candidates: list[str],
    settings: GenerativeProviderSettings,
    operation: str,
    ) -> list[str]:
    """Apply operation-specific candidate length filtering."""
    filtered = []
    for candidate in candidates:
        word_count = len(candidate.split())
        if operation == "tag" and not (settings.word_limits.tag_min <= word_count <= settings.word_limits.tag_max):
            continue
        if operation == "title" and word_count > settings.word_limits.title_max:
            continue
        if operation == "summarize" and word_count > settings.word_limits.summarize_max:
            continue
        filtered.append(candidate)

    return filtered


def _parsed_candidates(
    responses: list[str],
    operation: str,
    settings: GenerativeProviderSettings,
    ) -> list[str]:
    """Parse generated responses into clean candidate phrases."""
    pattern = settings.parse_patterns.outline_tag if operation in {"outline", "tag"} else settings.parse_patterns.default

    candidates = []
    for response in responses:
        if len(response) < settings.candidate_min_chars:
            continue

        parts = re.split(pattern, response)
        for part in parts:
            phrase = part.strip()
            if len(phrase) < settings.candidate_min_chars:
                continue
            if not re.search(r"[a-zA-Z]", phrase):
                continue
            candidates.append(" ".join(phrase.split()))

    return list(dict.fromkeys(candidates))


class DefaultAnalysisProvider(AnalysisProvider):
    """Analysis provider backed by sentiment, spam, and toxicity models."""
    def analyze(self, content: str) -> dict:
        settings = load_settings().provider.analysis

        sentiment_scores = get_sentiment_model()(content)
        polarity_score = float(get_polarity_model()(content)["score"])
        spam_score = float(get_spam_model()(content)["score"])
        toxicity_score = float(get_toxicity_model()(content)["score"])

        return {
            "negative": round(float(sentiment_scores["negative"]), settings.round_digits),
            "neutral": round(float(sentiment_scores["neutral"]), settings.round_digits),
            "positive": round(float(sentiment_scores["positive"]), settings.round_digits),
            "polarity": round(polarity_score, settings.round_digits),
            "spam": round(spam_score, settings.round_digits),
            "toxicity": round(toxicity_score, settings.round_digits),
        }


class DefaultExtractionProvider(ExtractionProvider):
    """Extraction provider backed by entity and keyword logic."""
    def extract(self, content: str) -> dict:
        settings = load_settings().provider.extraction

        doc_model = get_document_model()
        keyword_model = get_keyword_model(top_n=settings.keyword_top_n)

        entities = [entity.text.strip() for entity in doc_model(content).ents if entity.text.strip()]
        entity_candidates, entity_scores = semantic_similarity(content, entities)

        keywords = keyword_model(content)
        keyword_candidates, keyword_scores = semantic_similarity(content, keywords)

        return {
            "entities": {
                candidate: score
                for candidate, score in zip(entity_candidates[: settings.entity_limit], entity_scores[: settings.entity_limit])
            },
            "keywords": {
                candidate: score
                for candidate, score in zip(keyword_candidates[: settings.keyword_limit], keyword_scores[: settings.keyword_limit])
            },
        }


class DefaultGenerativeProvider(GenerativeProvider):
    """Generative provider using prompt templates and scored candidate parsing."""
    def generate(self, content: str, operation: str) -> dict:
        settings = load_settings().provider.generative
        prompts = settings.prompt_list(operation)
        if not prompts:
            raise ValueError(f"Unsupported generation operation: {operation}")

        generator = get_generative_model()
        delimiter = settings.delimiter.for_operation(operation)

        all_candidates = []
        for prompt in prompts:
            text_prompt = settings.default_template.format(prompt=prompt, content=content, delimiter=delimiter)
            responses = generator(text_prompt)
            all_candidates.extend(_parsed_candidates(responses, operation=operation, settings=settings))

        filtered_candidates = _candidate_filters(all_candidates, settings=settings, operation=operation)
        ranked_candidates, ranked_scores = composite_scores(content, filtered_candidates)

        result = {}
        for candidate, score in zip(ranked_candidates[: settings.result_limit], ranked_scores[: settings.result_limit]):
            result[candidate] = float(score)

        return result


class DefaultRelevanceProvider(RelevanceProvider):
    """Relevance provider backed by semantic similarity scoring."""
    def score(self, content: str, candidates: list[str]) -> dict:
        ranked_candidates, ranked_scores = semantic_similarity(content, candidates)
        return {
            candidate: float(score)
            for candidate, score in zip(ranked_candidates, ranked_scores)
        }
