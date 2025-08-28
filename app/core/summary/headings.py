import spacy

from app.core.metrics.categories import classify_content
from app.core.summary.generate import generate_summary
from app.core.tags.similarity import semantic_similarity

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
        "Generate many short conise single-setence descriptions of the following text",
        "List several brief explanations of the following text",
        "List multiple varied summaries of the following text",
        "Rephrase the following terms into a list of concise pithy summaries"
    ],
    "outline": [
        "In 6 words or less explain the main point of the following text",
        "In as few words as possible, list the key points of the following text",
        "In as few words as possible, describe the idea of the following text",
        "Rephrase the following terms into a list of concise very short descriptions"
    ]
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


# NOTE: Perform map-reduce summarization over content sentences
def get_outline(content: str, top_n=3) -> tuple[list, str]:
    """Perform map-reduce sentence summarization to generate an outline"""
    sentence_headings = []
    for sentence in spacy_nlp(content).sents:
        # Initialize a list of blank headings to ensure that top_n values exist
        headings = get_headings(sentence.text, heading="outline", top_n=top_n)
        sentence_headings.append(headings)

    generation_kwargs = dict(format="list", max_new_tokens=top_n * 4, temperature=0.7)
    candidates = []
    prompt = "Select the minimum set of non-overalpping key points from the following:"    
    # Combine all sentence headings into a fluid continous outline
    for i in range(top_n):
        heading_strings = " ".join([headings[i] for headings in sentence_headings if len(headings) > i])
        candidates += generate_summary(content=heading_strings, prompt=prompt, **generation_kwargs)

    selected_outline = semantic_similarity(content, candidates, top_n=top_n)

    return selected_outline


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
    for heading in HEADING_PROMPTS.keys():
        result = get_headings(sample_text, heading=heading, top_n=5)
        print(f"\nGenerated {heading}s:", result)

    print("\n=== Generate Outline ===")


if __name__ == "__main__":
    demo_headings()
