from app.core.summary.generate import generate_summary
from app.core.utils.relevance import maximal_marginal_relevance, semantic_similarity
from app.core.utils.samples import SAMPLE_TEXT

from app.config import get_settings
from app.core.utils.models import get_models


# Define module level constants
settings = get_settings()
TAG_PROMPTS = settings.defaults.tags

# Get module level variables
models = get_models()
embedding_model = models['embedding_model']
key_bert = models['key_bert']
spacy_nlp = models['spacy_nlp']
yake_extractor = models['yake_extractor']


def extract_entities(content: str, top_n: int=5) -> list:
    """Extract entities and return the top_n results"""
    # Extract entity tags from spacy pipeline
    entities = list({entity.text.strip() for entity in spacy_nlp(content).ents})

    if len(entities):
        entities, scores = semantic_similarity(content, entities)
        return entities[:top_n], scores[:top_n]
    else:
        return [], []


def extract_keywords(content: str, top_n: int=10) -> list:
    """Extract entities and return the top_n most relevant results"""
    # Bert keywords
    bert_keywords = key_bert.extract_keywords(
        content, 
        keyphrase_ngram_range=(1, 1), 
        stop_words='english', 
        top_n=top_n, 
        use_mmr=False
    )
    bert_keywords = [phrase for phrase, score in bert_keywords]
   
    # Yake keywords (adds recall)
    yake_keywords = yake_extractor.extract_keywords(content)
    yake_keywords = [phrase for phrase, score in yake_keywords]

    # Combine all keywords and compare source relevance with cosine similiarty
    candidates = []
    for keys in (bert_keywords, yake_keywords):
        candidates.extend([k.lower() for k in keys])

    candidates = list(set(candidates))
    keywords, scores = semantic_similarity(content, candidates)

    return keywords[:top_n], scores[:top_n]


def extract_related(content: str, min_length: int=1, max_length: int=3, top_n: int=10) -> dict:
    """Use language models to generate lists of related concepts and topics"""
    # Generate related topics, themes, and concepts
    tag_strings = []
    for prompt in TAG_PROMPTS:
        tag_strings += generate_summary(
            content=content,
            prompt=prompt,
            format="list", 
            max_new_tokens=top_n * 12, 
            temperature=0.7
        )

    # Filter generated tag by length and return the top_n similarity results    
    candidates = [s for s in tag_strings if len(s.split()) >= min_length and len(s.split()) <= max_length]
    tags, scores = maximal_marginal_relevance(content, candidates)

    return tags[:top_n], scores[:top_n]


# Example usage and testing function
def demo_tagger():
    """Test the tagging functionality with different parameters."""
    print("\n=== Basic Tagging ===")
    result = extract_entities(SAMPLE_TEXT, top_n=5)
    print("\nExtracted entities:", result)

    result = extract_keywords(SAMPLE_TEXT, top_n=8)
    print("\nExtracted keywords:", result)

    result = extract_related(SAMPLE_TEXT, min_length=1, max_length=3, top_n=5)
    print("\nExtracted related tags (min ngram=1):", result)

    result = extract_related(SAMPLE_TEXT, min_length=2, max_length=5, top_n=5)
    print("\nExtracted related tags (min ngram=2):", result)


if __name__ == "__main__":
    demo_tagger()
