from app.core.summary.generate import generate_summary
from app.core.tags.similarity import semantic_similarity

# Define module-level constants
TITLE_PROMPTS = [
    "In 5 words or less, generate several concise and engaging titles about the following text",
    "In as few words as possible, list several short, attention grabbing titles for the following text",
    "In as few words as possible, list various potential headlines related to the following text"
]


def get_titles(content: str, top_n: int=5) -> tuple[list, str]:
    """Extract entities and return the top_n results"""
    # Generate multiple candidate labels
    candidates = []
    for prompt in TITLE_PROMPTS:
        # Generate candidate tiles for each prompt
        candidates += generate_summary(
            content=content, 
            prompt=prompt, 
            format="list", 
            max_new_tokens=top_n * 4, 
            temperature=0.7
        )

    # Combine all generated titles into a new content string
    title_content = " ".join(candidates)

    # Add some more variety detached from the source content
    prompt = "Rephrase the following terms into a list of short pithy titles"
    candidates += generate_summary(
        content=title_content, 
        prompt=prompt, 
        format="list", 
        max_new_tokens=top_n * 4,
        temperature=0.7
    )
    candidates = list({s.lower() for s in candidates})

    # Select by similarity among generated titles
    # NOTE: This *should* filter out unrelated model banter
    selected = semantic_similarity(title_content, candidates, top_n=top_n)

    return selected


# Example usage and testing function
def demo_extractor():
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
    
    print("\n=== Generate Formatted Summaries ===")
    result = get_titles(sample_text, top_n=5)
    print("\nGenerated Titles:", result)


if __name__ == "__main__":
    demo_extractor()
