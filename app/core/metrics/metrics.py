import numpy

from textblob import TextBlob
from transformers import pipeline

# Define module-level variables
zero_shot_pipe = pipeline(model='facebook/bart-large-mnli')


def classify_content(content: str, labels: list) -> list:
    """Return the zero-shot classification scores in the order of the supplied labels"""
    result = zero_shot_pipe(content, candidate_labels=labels)
    scores = {label: score for label, score in zip(result['labels'], result['scores'])} 

    # Return scores in the order the labels were provided
    return [scores[k] for k in labels]


def content_subjectivity(content: str) -> list:
    """Return the zero-shot classification scores for subjectivity (tone)"""
    pass


def content_granularity(content: str) -> list:
    """Return the zero-shot classification scores for granularity (detail)"""
    pass


# Example usage and testing function
def demo_metrics():
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
    neutral_text = "AI may be a devistating force, it could possibly mean the end of the world"
    positive_text = "AI will bring about a utopian revolution, it will be very beneficial"

    print("\n== Document Metrics ===")

    for label, content in zip(content_labels, (negative_text, neutral_text, positive_text, sample_text)):
        print(f"\nText: {label}")
        
        blob_score = TextBlob(content).sentiment.subjectivity
        print("Blob_score:", round(blob_score, 4))

        # Subjectivity score
        class_labels = ['dogmatic', 'subjective', 'balanced', 'objective', 'impartial']
        result = classify_content(content, class_labels)
        print("Subjectivity:", [round(v, 4) for v in result], "Label:", class_labels[numpy.argmax(result)])

        # Granularity score
        class_labels = ['generic', 'limited', 'concise', 'detailed', 'nuanced']
        result = classify_content(content, class_labels)
        print("Granularity:", [round(v, 4) for v in result], "Label:", class_labels[numpy.argmax(result)])


if __name__ == "__main__":
    demo_metrics()
