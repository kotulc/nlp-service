import spacy
import yake

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT


# Define module level variables
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
key_bert = KeyBERT('all-MiniLM-L6-v2')
spacy_nlp = spacy.load("en_core_web_lg")
yake_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=20, features=None)


def semantic_similarity(text_body, word_list, top_n=10):
    """Rank words by semantic similarity to text using embeddings"""
    # Create embeddings
    text_embedding = embedding_model.encode([text_body])
    word_embeddings = embedding_model.encode(word_list)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(text_embedding, word_embeddings)[0]
    
    # Create word-score pairs
    word_scores = list(zip(word_list, similarities))
    
    return sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_n]


# Example usage and testing function
def demo_tagger():
    """Test the tagging functionality with different parameters."""

    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to 
    natural intelligence displayed by animals including humans. Leading AI textbooks define 
    the field as the study of "intelligent agents": any system that perceives its environment 
    and takes actions that maximize its chance of achieving its goals. Some popular accounts 
    use the term "artificial intelligence" to describe machines that mimic "cognitive" 
    functions that humans associate with the human mind, such as "learning" and "problem solving".
    As machines become increasingly capable, tasks considered to require "intelligence" are 
    often removed from the definition of AI, a phenomenon known as the AI effect. For instance, 
    optical character recognition is frequently excluded from things considered to be AI, 
    having become a routine technology.
    """
    
    print("=== Basic Tagging ===")
    bert_phrases = key_bert.extract_keywords(sample_text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10, use_mmr=False)
    bert_phrases = [phrase for phrase, score in bert_phrases]
    print("\nKeyBERT Keywords:", bert_phrases)

    # Spacy entity detection
    doc = spacy_nlp(sample_text)
    spacy_entities = list({entity.text.strip() for entity in doc.ents})
    print("\nEntities:", spacy_entities)

    # YAKE keywords (adds recall)
    yake_phrases = yake_extractor.extract_keywords(sample_text)
    yake_phrases = [phrase for phrase, score in yake_phrases]
    print("\nYAKE keywords:", yake_phrases)

    candidates = []
    for keys in (bert_phrases, yake_phrases, spacy_entities):
        candidates.extend([k.lower() for k in keys])
    candidates = list(set(candidates))
    print("\nCandidate Keywords:", candidates)

    ranked_keywords = semantic_similarity(sample_text, candidates)
    print("\nRanked by Semantic Similarity:", ranked_keywords)


if __name__ == "__main__":
    demo_tagger()
