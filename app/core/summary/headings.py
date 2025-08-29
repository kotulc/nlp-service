import numpy
import spacy

from app.core.metrics.categories import classify_content
from app.core.summary.generate import generate_summary
from app.core.tags.similarity import semantic_similarity

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Define module level variables
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define module-level constants
HEADING_PROMPTS = {
    "title": [
        "In 6 words or less, generate multiple concise and engaging titles for the following text",
        "In as few words as possible, list several short, attention grabbing titles for the following text",
        "In as few words as possible, list various potential headlines related to the following text",
        "Rephrase the following terms into a list of short pithy titles"
    ],
    "subtitle": [
        "Generate several terse and succinct phrases describing the following text",
        "list several short, 8 word maximum attention grabbing captions for the following text",
        "In as few words as possible, explain the following text",
        "Rephrase the following terms into a list of short pithy subtitles"
    ],
    "description": [
        "Generate many short concise single-sentence descriptions of the following text",
        "List several brief, terse explanations of the following text",
        "List multiple varied summaries of the following text",
        "Rephrase the following terms into a list of concise summaries"
    ],
}

# Define module-level variables
spacy_nlp = spacy.load("en_core_web_lg")


def get_headings(content: str, heading="title", top_n: int=5) -> tuple[list, str]:
    """Generate a list of short heading summaries for the supplied content"""
    if not heading in HEADING_PROMPTS:
        raise ValueError(f"Supplied heading type '{heading}' is not a supported value.")
    
    # Define common generation kwargs
    generation_kwargs = dict(format="list", max_new_tokens=top_n * 4, temperature=0.7)

    candidates = []
    for prompt in HEADING_PROMPTS[heading][:-1]:
        # Generate candidate tiles for each prompt
        candidates += generate_summary(content=content, prompt=prompt, **generation_kwargs)

    # Combine all generated titles into a new content string
    title_content = " ".join(candidates)

    # Add some more variety detached from the source content
    re_prompt = HEADING_PROMPTS[heading][-1]
    candidates += generate_summary(content=title_content, prompt=re_prompt, **generation_kwargs)
    candidates = list({s.lower() for s in candidates})

    # Select by similarity among generated titles
    # NOTE: This *should* filter out unrelated model banter
    selected_titles = semantic_similarity(title_content, candidates, top_n=top_n)

    return selected_titles


def get_outline(content: str, similarity_threshold=0.7, top_n=3) -> tuple[list, str]:
    """Perform map-reduce sentence summarization to generate an outline"""
    # Split the supplied content string into individual sentences
    content_sections = [s.text for s in spacy_nlp(content).sents]

    if len(content_sections) == 0:
        raise ValueError("Supplied content string must contain one or more sentences.")

    # Compare similarity among all sentences
    content_embedding = embedding_model.encode(content_sections)
    similarities = cosine_similarity(content_embedding)
    
    # TODO: Test other methods of reducing sections
    # Combine sentences with similarity above a given threshold
    content_reduced = [content_sections[0]]
    i, j = 0, 1
    while j < similarities.shape[0]:
        if similarities[i, j] >= similarity_threshold:
            content_reduced[i] += " " + content_sections[j]
        else:
            content_reduced.append(content_sections[j])
            i = j
        j += 1

    # Generate candidate section descriptors
    generation_kwargs = dict(format="list", max_new_tokens=32, temperature=0.7)

    section_descriptions = []
    for section in content_reduced:
        # Generate candidate summaries for each prompt
        candidates = []
        for prompt in HEADING_PROMPTS["description"][:-1]:
            candidates += generate_summary(content=section, prompt=prompt, **generation_kwargs)
        
        # Compute section embeddings
        content_embedding = embedding_model.encode([section])
        candidate_embeddings = embedding_model.encode(candidates)

        # TODO: Select top_n best candidates sorted by similarity
        # Select the candidate with the highest similarity
        scores = cosine_similarity(content_embedding, candidate_embeddings).flatten()
        selected_candidate = candidates[numpy.argmax(scores)]
        section_descriptions.append(selected_candidate)

    return section_descriptions


# Example usage and testing function
def demo_headings():
    """Test the heading generation functionality"""

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
    
    print("\n=== Generate Headings ===")
    # for heading in HEADING_PROMPTS.keys():
    #     result = get_headings(sample_text, heading=heading, top_n=5)
    #     print(f"\nGenerated {heading}s:", result)

    print("\n=== Generate Outline ===")
    result = get_outline(sample_text)


if __name__ == "__main__":
    demo_headings()
