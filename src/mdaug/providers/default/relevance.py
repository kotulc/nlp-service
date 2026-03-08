"""Default-provider adapters that apply configured models to core relevance algorithms."""

from mdaug.core.relevance.relevance import (
    composite_scores as _core_composite_scores,
    maximal_marginal_relevance as _core_maximal_marginal_relevance,
    semantic_similarity as _core_semantic_similarity,
)
from mdaug.providers.default.models import get_acceptability_model, get_embedding_model
from mdaug.providers.default.settings import load_settings


def _acceptability_score(content: str) -> float:
    """Return acceptability score for content using configured default model."""
    model = get_acceptability_model()
    return float(model(content)["score"])


def composite_scores(content: str, candidates: list[str]) -> tuple[list[str], list[float]]:
    """Score candidates by similarity times linguistic acceptability."""
    settings = load_settings().relevance
    return _core_composite_scores(
        content,
        candidates,
        encode=get_embedding_model(),
        score_acceptability=_acceptability_score,
        semantic_score_precision=settings.semantic.score_precision,
        score_precision=settings.composite.score_precision,
    )


def maximal_marginal_relevance(
    content: str,
    candidates: list[str],
    sim_lambda: float | None = None,
    top_n: int | None = None,
    ) -> tuple[list[str], list[float]]:
    """Select candidates balancing query relevance and inter-candidate diversity."""
    settings = load_settings().relevance.mmr
    selected_lambda = settings.sim_lambda if sim_lambda is None else sim_lambda
    selected_top_n = settings.top_n if top_n is None else top_n

    return _core_maximal_marginal_relevance(
        content,
        candidates,
        encode=get_embedding_model(),
        sim_lambda=selected_lambda,
        top_n=selected_top_n,
        score_precision=settings.score_precision,
    )


def semantic_similarity(content: str, candidates: list[str]) -> tuple[list[str], list[float]]:
    """Rank candidates by semantic similarity to content embeddings."""
    settings = load_settings().relevance.semantic
    return _core_semantic_similarity(
        content,
        candidates,
        encode=get_embedding_model(),
        score_precision=settings.score_precision,
    )
