import numpy

from textblob import TextBlob
from transformers import pipeline

# Define elements of literary analysis
# Attempt to define labels on a scale of very low to very high in a given linguistic quality
DICTION_LABELS = ['formal', 'concrete', 'informal', 'colloquial', 'literary', 'poetic', 'abstract']
GENRE_LABELS = ['romance', 'drama', 'suspense', 'historical', 'non-fiction', 'adventure', 'sci-fi', 'fantasy']
STYLE_LABELS = ['expository', 'descriptive', 'persuasive', 'narrative', 'creative', 'experimental']
TONE_LABELS = ['dogmatic', 'subjective', 'neutral', 'objective', 'impartial']

# Define module-level variables
zero_shot_pipe = pipeline(model='facebook/bart-large-mnli')


def classify_content(content: str, labels: list, multi_label=False) -> list:
    """Return the zero-shot classification scores in the order of the supplied labels"""
    # NOTE: If more than one label can be correct, set multi_label=True
    result = zero_shot_pipe(content, candidate_labels=labels, multi_label=multi_label)
    scores = {label: score for label, score in zip(result['labels'], result['scores'])} 

    # Return scores in the order the labels were provided
    return [scores[k] for k in labels]


def score_diction(content: str) -> tuple[(list, str)]:
    """Return the zero-shot classification scores for diction"""
    # Zero-shot diction score (ideally this uses a fine-tuned a model)
    result = classify_content(content, DICTION_LABELS)
    return [round(float(v), 4) for v in result], DICTION_LABELS[numpy.argmax(result)]


def score_genre(content: str) -> tuple[(list, str)]:
    """Return the zero-shot classification scores for genre"""
    # Zero-shot genre score (ideally this uses a fine-tuned a model)
    result = classify_content(content, GENRE_LABELS)
    return [round(float(v), 4) for v in result], GENRE_LABELS[numpy.argmax(result)]


def score_style(content: str) -> tuple[(list, str)]:
    """Return the zero-shot classification scores for style"""
    # Zero-shot style score (ideally this uses a fine-tuned a model)
    result = classify_content(content, STYLE_LABELS)
    return [round(float(v), 4) for v in result], STYLE_LABELS[numpy.argmax(result)]


def score_tone(content: str) -> tuple[(list, str)]:
    """Return the zero-shot classification and textblob scores for subjectivity (tone)"""
    # Textblob subjectvitiy scores range [0.0, 1.0] with 1.0 being highly subjective
    blob_score = TextBlob(content).sentiment.subjectivity

    # Find distances from value to each label 'bucket'
    buckets = numpy.linspace(0, 1, len(TONE_LABELS))
    distances = numpy.abs(numpy.array(buckets) - blob_score)
    
    # Calculate gaussian kernel weights and normalize to sum to 1
    weights = numpy.exp(-1 * (distances**2) / 0.1)
    distribution = weights / numpy.sum(weights)

    # Zero-shot subjectivity score (ideally this uses a fine-tuned a model)
    result = classify_content(content, TONE_LABELS)

    # Combine scores and re-normalize
    result = (result + distribution) / numpy.sum(result + distribution)
    return [round(float(v), 4) for v in result], TONE_LABELS[numpy.argmax(result)]


# Example usage and testing function
def demo_categories():
    """Test the summarization function with different parameters"""

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

    content_labels = ('negative', 'neutral', 'positive', 'document')
    negative_text = "I hate AI, it is the worst thing to happen to humanity ever"
    neutral_text = "AI may be a devistating force, it could cause great harm if we don't guide it properly"
    positive_text = "AI will bring about a utopian revolution, it will be very beneficial"

    print("\n== Document Analysis ===")
    for content_label, content in zip(content_labels, (negative_text, neutral_text, positive_text, sample_text)):
        print(f"\nText: {content_label}")
        
        # Diction score and label
        scores, label = score_diction(content)
        print("Diction score:", scores, "Label:", label)

        # Genre score and label
        scores, label = score_genre(content)
        print("Genre score:", scores, "Label:", label)

        # Style score and label
        scores, label = score_style(content)
        print("Style score:", scores, "Label:", label)

        # Subjectivity score and label
        scores, label = score_tone(content)
        print("Tone score:", scores, "Label:", label)


if __name__ == "__main__":
    demo_categories()
