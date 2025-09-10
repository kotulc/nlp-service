import numpy

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load a model fine-tuned on the CoLA dataset fot linguistic acceptability scoring
classifier = pipeline("text-classification", model="textattack/roberta-base-CoLA")

# Define module level variables
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def composite_selection(content: str, candidates: list[str], top_n: int=3) -> list:
    """Select candidates using compound (linguistic + similarity) scores"""
    # Filter out duplicate candidate headings
    candidates = list({s.lower() for s in candidates})

    # Calculate linguistic acceptability scores for each candidate
    linguistic_scores = numpy.array([classifier(candidate)[0]['score'] for candidate in candidates])

    # Select the candidate with the highest compound (content similarity * linguistic) scores
    content_embedding = embedding_model.encode([content])
    candidate_embeddings = embedding_model.encode(candidates)
    similarity_scores = cosine_similarity(content_embedding, candidate_embeddings).flatten()
    candidate_scores = list(zip(candidates, linguistic_scores * similarity_scores))
    sorted_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)

    return [candidate for candidate, _ in sorted_scores][:top_n]
