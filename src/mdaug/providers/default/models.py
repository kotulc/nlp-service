"""Lazy model loaders used by default non-mock providers."""

from functools import lru_cache
from typing import Any

from mdaug.providers.default.settings import load_settings


def _hashable_kwargs(values: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Normalize generation kwargs into a cache-safe tuple representation."""
    pairs = []
    for key, value in values.items():
        if not isinstance(key, str):
            continue

        if isinstance(value, (bool, float, int, str)) or value is None:
            pairs.append((key, value))

    return tuple(sorted(pairs, key=lambda item: item[0]))


def _unwrap_classification(result) -> list[dict]:
    """Normalize transformers classification outputs into a list of dict items."""
    if isinstance(result, list) and result and isinstance(result[0], list):
        return result[0]

    if isinstance(result, list):
        return result

    if isinstance(result, dict):
        return [result]

    return []


@lru_cache(maxsize=8)
def _load_acceptability_model(model_name: str):
    """Load an acceptability scoring callable for a configured model name."""
    try:
        from transformers import pipeline

        classifier = pipeline("text-classification", model=model_name)
    except Exception as exc:
        raise RuntimeError(f"Unable to load acceptability model '{model_name}': {exc}") from exc

    def score_acceptability(content: str) -> dict[str, float]:
        """Compute acceptability score for the supplied string."""
        result = classifier(content, truncation=True)
        return {"score": float(result[0]["score"])}

    return score_acceptability


def get_acceptability_model():
    """Return a CoLA-style acceptability scoring callable."""
    settings = load_settings().models.acceptability
    return _load_acceptability_model(settings.model_name)


@lru_cache(maxsize=8)
def _load_document_model(model_names: tuple[str, ...]):
    """Load a configured spaCy model for sentence/entity extraction."""
    try:
        import spacy
    except ModuleNotFoundError as exc:
        raise RuntimeError("spaCy is required for the default document model.") from exc

    for model_name in model_names:
        try:
            return spacy.load(model_name)
        except OSError:
            continue

    names = ", ".join(model_names)
    raise RuntimeError(f"Unable to load any configured spaCy model: {names}")


def get_document_model():
    """Return a configured spaCy model for sentence/entity extraction."""
    settings = load_settings().models.document
    return _load_document_model(settings.model_names)


@lru_cache(maxsize=8)
def _load_embedding_model(model_name: str):
    """Load a configured sentence embedding encoder callable."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(f"Unable to load embedding model '{model_name}': {exc}") from exc

    return model.encode


def get_embedding_model():
    """Return a sentence-transformers embedding encoder callable."""
    settings = load_settings().models.embedding
    return _load_embedding_model(settings.model_name)


@lru_cache(maxsize=8)
def _load_generative_model(model_name: str, default_kwargs: tuple[tuple[str, Any], ...]):
    """Load a configured transformers text generation callable."""
    try:
        from transformers import pipeline

        generator = pipeline("text-generation", model=model_name)
    except Exception as exc:
        raise RuntimeError(f"Unable to load generative model '{model_name}': {exc}") from exc

    base_kwargs = dict(default_kwargs)

    def infer(content: str, **kwargs) -> list[str]:
        """Return generated text from the configured language model."""
        model_kwargs = base_kwargs.copy()
        model_kwargs.update(kwargs)
        sequences = generator(content, return_full_text=False, **model_kwargs)
        return [str(sequence.get("generated_text", "")).strip() for sequence in sequences]

    return infer


def get_generative_model(model_name: str | None = None):
    """Return a text generation callable using configured or explicit model names."""
    settings = load_settings().models.generative
    selected_name = model_name if isinstance(model_name, str) and model_name else settings.model_name
    default_kwargs = _hashable_kwargs(settings.kwargs)
    return _load_generative_model(selected_name, default_kwargs)


@lru_cache(maxsize=16)
def _load_keyword_model(
    top_n: int,
    embedding_model_name: str,
    keyphrase_ngram_min: int,
    keyphrase_ngram_max: int,
    stop_words: str,
    use_mmr: bool,
    yake_language: str,
    yake_ngram_size: int,
    yake_dedup_limit: float,
    yake_dedup_func: str,
):
    """Load a configured KeyBERT + YAKE keyword extraction callable."""
    half_top_n = max(1, top_n // 2)

    try:
        import keybert
        import yake

        keybert_model = keybert.KeyBERT(embedding_model_name)
        yake_extractor = yake.KeywordExtractor(
            lan=yake_language,
            n=yake_ngram_size,
            dedupLim=yake_dedup_limit,
            dedupFunc=yake_dedup_func,
            top=half_top_n,
            features=None,
        )
    except Exception as exc:
        raise RuntimeError("Unable to initialize keyword extraction models.") from exc

    def extract_keywords(content: str) -> list[str]:
        """Extract and combine KeyBERT and YAKE keyword candidates."""
        keybert_pairs = keybert_model.extract_keywords(
            content,
            keyphrase_ngram_range=(keyphrase_ngram_min, keyphrase_ngram_max),
            stop_words=stop_words,
            top_n=half_top_n,
            use_mmr=use_mmr,
        )
        keybert_keywords = [phrase for phrase, _score in keybert_pairs]
        yake_keywords = [phrase for phrase, _score in yake_extractor.extract_keywords(content)]
        unique_keywords = list(dict.fromkeys(keybert_keywords + yake_keywords))
        return [keyword.lower() for keyword in unique_keywords if keyword]

    return extract_keywords


def get_keyword_model(top_n: int | None = None):
    """Return a combined KeyBERT + YAKE keyword extraction callable."""
    settings = load_settings().models.keyword
    selected_top_n = top_n if isinstance(top_n, int) and top_n > 0 else settings.default_top_n

    return _load_keyword_model(
        selected_top_n,
        settings.embedding_model_name,
        settings.keyphrase_ngram_min,
        settings.keyphrase_ngram_max,
        settings.stop_words,
        settings.use_mmr,
        settings.yake_language,
        settings.yake_ngram_size,
        settings.yake_dedup_limit,
        settings.yake_dedup_func,
    )


@lru_cache(maxsize=1)
def _load_polarity_model():
    """Load a polarity scoring callable using TextBlob and VADER."""
    try:
        from textblob import TextBlob
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception as exc:
        raise RuntimeError("TextBlob and VADER are required for polarity scoring.") from exc

    analyzer = SentimentIntensityAnalyzer()

    def score_polarity(content: str) -> dict[str, float]:
        """Compute polarity score by averaging TextBlob and VADER compound polarity."""
        blob_score = float(TextBlob(content).sentiment.polarity)
        vader_score = float(analyzer.polarity_scores(content)["compound"])
        return {"score": (blob_score + vader_score) / 2}

    return score_polarity


def get_polarity_model():
    """Return a polarity scoring callable using TextBlob and VADER."""
    return _load_polarity_model()


@lru_cache(maxsize=1)
def _load_sentiment_model():
    """Load sentiment class scoring callable using VADER."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception as exc:
        raise RuntimeError("vaderSentiment is required for sentiment scoring.") from exc

    analyzer = SentimentIntensityAnalyzer()

    def score_sentiment(content: str) -> dict[str, float]:
        """Compute negative/neutral/positive sentiment scores."""
        sentiment = analyzer.polarity_scores(content)
        return {
            "negative": float(sentiment["neg"]),
            "neutral": float(sentiment["neu"]),
            "positive": float(sentiment["pos"]),
        }

    return score_sentiment


def get_sentiment_model():
    """Return sentiment class scoring callable using VADER."""
    return _load_sentiment_model()


@lru_cache(maxsize=8)
def _load_toxicity_model(model_name: str, top_k: int | None):
    """Load toxicity scorer backed by a transformer model."""
    try:
        from transformers import pipeline

        classifier = pipeline("text-classification", model=model_name, top_k=top_k)
    except Exception as exc:
        raise RuntimeError(f"Unable to load toxicity model '{model_name}': {exc}") from exc

    def score_toxicity(content: str) -> dict[str, float]:
        """Compute toxicity score as probability-like value in [0.0, 1.0]."""
        predictions = _unwrap_classification(classifier(content, truncation=True))
        by_label = {
            str(item.get("label", "")).lower(): float(item.get("score", 0.0))
            for item in predictions
        }
        if "toxic" in by_label:
            return {"score": round(by_label["toxic"], 4)}

        return {"score": round(max(by_label.values(), default=0.0), 4)}

    return score_toxicity


def get_toxicity_model():
    """Return a toxicity scoring callable backed by a transformer model."""
    settings = load_settings().models.toxicity
    top_k = settings.top_k if settings.top_k and settings.top_k > 0 else None
    return _load_toxicity_model(settings.model_name, top_k)


@lru_cache(maxsize=8)
def _load_spam_model(model_name: str, default_spam_index: int):
    """Load spam scorer backed by a transformer classification model."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("torch and transformers are required for spam scoring.") from exc

    try:
        spam_tokenizer = AutoTokenizer.from_pretrained(model_name)
        spam_classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(f"Unable to load spam model '{model_name}': {exc}") from exc

    spam_index = default_spam_index
    labels = {
        int(index): str(label).lower()
        for index, label in spam_classifier.config.id2label.items()
    }
    matched = [index for index, label in labels.items() if "spam" in label]
    if matched:
        spam_index = matched[0]

    def score_spam(content: str) -> dict[str, float]:
        """Compute spam score as probability-like value in [0.0, 1.0]."""
        inputs = spam_tokenizer(content, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = spam_classifier(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1).flatten()
        return {"score": round(float(probabilities[spam_index].item()), 4)}

    return score_spam


def get_spam_model():
    """Return a spam scoring callable backed by a transformer model."""
    settings = load_settings().models.spam
    return _load_spam_model(settings.model_name, settings.default_spam_index)
