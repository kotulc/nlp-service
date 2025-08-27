import spacy
import yake

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from app.core.summary.generate import generate_summary
from app.core.tags.similarity import maximal_marginal_relevance, semantic_similarity

# Define module level constants
TAG_PROMPTS = [
    "With as few words as possible, list several related trending topics from the following text",
    "With as few words as possible, list high level ideas and themes of the following text",
    "With as few words as possible, list several tangentially related concepts to the following text",
]

# Define module level variables
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
key_bert = KeyBERT('all-MiniLM-L6-v2')
spacy_nlp = spacy.load("en_core_web_lg")
yake_extractor = yake.KeywordExtractor(
    lan="en", 
    n=1, 
    dedupLim=0.9, 
    dedupFunc="seqm", 
    top=20, 
    features=None
)


def extract_entities(content: str, top_n: int=5) -> list:
    """Extract entities and return the top_n results"""
    # Extract entity tags from spacy pipeline
    return list({entity.text.strip() for entity in spacy_nlp(content).ents})[:top_n]


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
    ranked_keywords = semantic_similarity(content, candidates, top_n=top_n)

    return ranked_keywords


def extract_tags(content: str, min_length: int=1, max_length: int=3, top_n: int=10) -> dict:
    """Use language models to generate lists of related concepts and topics"""
    # Generate related topics, themes, and concepts
    tag_strings = []
    for prompt in TAG_PROMPTS:
        tag_strings += generate_summary(
            content=content,
            prompt=prompt,
            format="list", 
            max_new_tokens=top_n * 4, 
            temperature=0.7
        )

    # Filter generated tag by length and return the top_n similarity results    
    candidates = [s for s in tag_strings if len(s.split()) >= min_length and len(s.split()) <= max_length]
    maximal_tags = maximal_marginal_relevance(content, candidates, top_n=top_n)

    return maximal_tags


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
    
    print("\n=== Basic Tagging ===")
    result = extract_entities(sample_text, top_n=5)
    print("\nExtracted entities:", result)

    result = extract_keywords(sample_text, top_n=8)
    print("\nExtracted keywords:", result)

    result = extract_tags(sample_text, min_length=1, max_length=3, top_n=5)
    print("\nExtracted tags (min ngram=1):", result)

    result = extract_tags(sample_text, min_length=2, max_length=5, top_n=5)
    print("\nExtracted tags (min ngram=2):", result)


if __name__ == "__main__":
    demo_tagger()
