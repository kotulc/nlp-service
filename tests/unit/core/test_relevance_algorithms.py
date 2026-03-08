"""Unit tests for provider-agnostic core relevance scoring algorithms."""

from mdaug.core.relevance.relevance import (
    composite_scores,
    maximal_marginal_relevance,
    semantic_similarity,
)


def _encode(vectors: dict[str, list[float]]):
    """Build a deterministic encoder callable from a text-to-vector mapping."""
    def encode(items: list[str]) -> list[list[float]]:
        return [vectors[item] for item in items]

    return encode


def test_semantic_similarity_orders_candidates_by_descending_similarity():
    """Semantic similarity ranks closest vectors first with rounded scores."""
    encode = _encode(
        {
            "query": [1.0, 0.0],
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
            "c": [0.8, 0.6],
        }
    )

    ranked, scores = semantic_similarity("query", ["b", "c", "a"], encode=encode, score_precision=3)

    assert ranked == ["a", "c", "b"]
    assert scores == [1.0, 0.8, 0.0]


def test_composite_scores_applies_acceptability_weighting_after_semantic_rank():
    """Composite scores apply acceptability weighting with candidate normalization."""
    encode = _encode(
        {
            "query": [1.0, 0.0],
            "a": [1.0, 0.0],
            "b": [0.8, 0.6],
        }
    )

    acceptability = {"a": 0.2, "b": 1.0}
    ranked, scores = composite_scores(
        "query",
        ["  A  ", "b"],
        encode=encode,
        score_acceptability=lambda candidate: acceptability[candidate],
        semantic_score_precision=3,
        score_precision=3,
    )

    assert ranked == ["b", "a"]
    assert scores == [0.8, 0.2]


def test_maximal_marginal_relevance_balances_relevance_against_diversity():
    """MMR selects diverse candidates when novelty penalty outweighs relevance gains."""
    encode = _encode(
        {
            "query": [1.0, 0.0],
            "a": [1.0, 0.0],
            "b": [0.9, 0.1],
            "c": [0.0, 1.0],
        }
    )

    ranked, scores = maximal_marginal_relevance(
        "query",
        ["a", "b", "c"],
        encode=encode,
        sim_lambda=0.2,
        top_n=2,
        score_precision=3,
    )

    assert ranked == ["a", "c"]
    assert scores == [1.0, 0.0]
