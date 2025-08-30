import numpy

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Define module level variables
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def maximal_marginal_relevance(content, candidates, sim_lambda=0.5, top_n=10) -> list:
    """Select candidate words using maximal marginal relevance scoring"""
    # Create embeddings
    content_embedding = embedding_model.encode([content])
    candidate_embeddings = embedding_model.encode(candidates)

    # Select the candidate with the highest similarity
    similarities = cosine_similarity(content_embedding, candidate_embeddings).flatten()

    # List available candidates (candidate, embedding, similarity)
    available_candidates = list(zip(candidates, candidate_embeddings, similarities))
    selected_index = numpy.argmax(similarities)
    selected = [available_candidates[selected_index]]
    available_candidates.pop(selected_index)

    for _ in range(top_n):
        if not len(available_candidates): break

        mmr_scores = []
        for _, embedding, similarity in available_candidates:
            # Calculate max similarities to selected items
            max_similarity = max([cosine_similarity(embedding.reshape(1, -1), emb.reshape(1, -1)).flatten() for _, emb, _ in selected])
            mmr_scores.append(sim_lambda * similarity - (1 - sim_lambda) * max_similarity)
        
        # Select the next best candidate and remove it from the pool
        selected_index = numpy.argmax(mmr_scores)
        selected.append(available_candidates[selected_index])
        available_candidates.pop(selected_index)

    return [candidate for candidate, _, _ in selected]


def semantic_similarity_selection(content: str, candidates: list, top_n: int=10) -> list:
    """Rank words by semantic similarity to text using embeddings"""
    # Create embeddings
    content_embedding = embedding_model.encode([content])
    candidate_embeddings = embedding_model.encode(candidates)
    
    # Calculate cosine similarity and create word-score pairs
    similarities = cosine_similarity(content_embedding, candidate_embeddings)[0]
    candidate_scores = list(zip(candidates, similarities))
    sorted_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)

    return [candidate for candidate, _ in sorted_scores][:top_n]
