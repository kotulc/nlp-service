import numpy
import spacy

from app.core.summary.generate import generate_summary

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Define module-level constants
HEADING_PROMPTS = {
    "title": [
        "In 5 words or less, list multiple concise and engaging titles for the following text",
        "In as few words as possible, list several short, attention grabbing titles for the following text",
        "In as few words as possible, list various potential headlines related to the following text",
        "Rephrase the following terms into a list of short pithy titles"
    ],
    "subtitle": [
        "state several terse and succinct statements describing the following text",
        "list several short, attention grabbing captions for the following text",
        "In 8 words or less, list various subtitles for the following text",
        "Rephrase the following statements into a list of short pithy subtitles"
    ],
    "description": [
        "Generate many short concise single-sentence descriptions of the following text",
        "List several brief, terse explanations of the following text",
        "List multiple varied summaries of the following text",
        "Rephrase the following statements into a list of concise summaries"
    ],
}

# Load a model fine-tuned on the CoLA dataset fot linguistic acceptability scoring
classifier = pipeline("text-classification", model="textattack/roberta-base-CoLA")

# Define module level variables
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define module-level variables
spacy_nlp = spacy.load("en_core_web_lg")


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


def get_headings(content: str, heading="title", top_n: int=3) -> tuple[list, str]:
    """Generate a list of short heading summaries for the supplied content"""
    if not heading in HEADING_PROMPTS:
        raise ValueError(f"Supplied heading type '{heading}' is not a supported value.")
    
    # Define common generation kwargs
    generation_kwargs = dict(format="list", max_new_tokens=top_n * 12, temperature=0.7)

    candidates = []
    for prompt in HEADING_PROMPTS[heading][:-1]:
        # Generate candidate tiles for each prompt
        candidates += generate_summary(content=content, prompt=prompt, **generation_kwargs)

    # Combine all generated titles into a new content string
    title_content = " ".join(candidates)

    # Add some more variety detached from the source content
    re_prompt = HEADING_PROMPTS[heading][-1]
    candidates += generate_summary(content=title_content, prompt=re_prompt, **generation_kwargs)

    return composite_selection(content, candidates, top_n)


def get_outline(content: str, n_sections=3, top_n=3) -> list[list]:
    """Perform map-reduce sentence summarization to generate an outline"""
    # Split the supplied content string into individual sentences
    content_sentences = [s.text for s in spacy_nlp(content).sents]

    if len(content_sentences) == 0:
        raise ValueError("Supplied content string must contain one or more sentences.")

    # Split the document into n_sections and iteratively combine sentences 
    sections = []
    n_sentences = int(numpy.ceil(len(content_sentences) / n_sections))
    for i in range(0, n_sentences * n_sections, n_sentences):
        sections.append(" ".join(content_sentences[i:i + n_sentences]))

    # TODO: Pull these args from configs
    # Define outline-specific generation kwargs
    generation_kwargs = dict(format="list", max_new_tokens=32, temperature=0.7)

    # Generate candidate section descriptions
    section_descriptions = []
    for section in sections:
        # Generate candidate summaries for each prompt
        section_candidates = []
        for prompt in HEADING_PROMPTS["description"][:-1]:
            section_candidates += generate_summary(content=section, prompt=prompt, **generation_kwargs)
        
        # Score and select the top_n descriptions for each section
        selections = composite_selection(content=section, candidates=section_candidates, top_n=top_n)
        section_descriptions.append(selections)

    return section_descriptions


# Example usage and testing function
def demo_headings():
    """Test the heading and section outline generation functionality"""

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
    n_sections, top_n = 3, 5
    for heading in HEADING_PROMPTS.keys():
        result = get_headings(sample_text, heading=heading, top_n=top_n)
        print(f"\nGenerated {heading}s:", result)

    print("\n=== Generate Outline ===")
    result = get_outline(sample_text, n_sections=n_sections, top_n=top_n)
    for i in range(n_sections):
        section_summaries = [section[i] for section in result if len(section) > i]
        print(f"\nOutline candidate {i + 1}:")
        for section_summary in section_summaries:
            print("\t-", section_summary)


if __name__ == "__main__":
    demo_headings()
