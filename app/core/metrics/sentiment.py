import numpy

from typing import Dict

from app.core.metrics.style import classify_content
from app.core.utils.samples import SAMPLE_TEXT, NEGATIVE_TEXT, NEUTRAL_TEXT, POSITIVE_TEXT
from app.core.utils.models import get_document_model, get_sentiment_model


# Define sentiment class constant
SENTIMENT_CLASSES = ["negative", "neutral", "positive"]

# Get module level variables
vader_analyzer = get_sentiment_model()
doc_model = get_document_model()


def content_sentiment(content: str) -> dict:
    """Compute bart and vader sentiment scores for the supplied string"""
    doc = doc_model(content)

    # Get bart and vader scores in an equivalent format (including precision)
    bart_scores = classify_content(doc.text, SENTIMENT_CLASSES)
    bart_scores = [round(score, 3) for score in bart_scores]
    vader_scores = vader_analyzer.polarity_scores(doc.text)
    vader_scores = [vader_scores[k] for k in ('neg', 'neu', 'pos')]

    return bart_scores, vader_scores


def sentence_sentiment(content: str) -> tuple[list]:
    """Compute bart and vader sentiment scores for each sentence in the supplied string"""
    doc = doc_model(content)

    sentence_list, bart_list, vader_list = [], [], []
    for sentence in doc.sents:
        # Get bart and vader scores in an equivalent format (including precision)
        sentence_list.append(sentence.text)

        bart_scores = classify_content(sentence.text, SENTIMENT_CLASSES)
        bart_scores = [round(score, 3) for score in bart_scores]
        bart_list.append(bart_scores)

        vader_scores = vader_analyzer.polarity_scores(sentence.text)
        vader_scores = [vader_scores[k] for k in ('neg', 'neu', 'pos')]
        vader_list.append(vader_scores)

    return sentence_list, bart_list, vader_list


def score_sentiment(content: str) -> Dict[str, float]:
    """Return the composite (mean) sentiment score for the supplied content"""
    # Composite score is the mean of two sentiment analyzers 
    bart_scores, vader_scores = content_sentiment(content)
    
    # Label scores and round to 4 decimal places
    scores = numpy.mean([bart_scores, vader_scores], axis=0)
    scores = dict(zip(SENTIMENT_CLASSES, [round(float(v), 4) for v in scores]))

    return scores


# Example usage and testing function
def demo_sentiment():
    """Test the sentiment functions with different parameters"""
    content_labels = ('negative', 'neutral', 'positive', 'document')
    content_text = (NEGATIVE_TEXT, NEUTRAL_TEXT, POSITIVE_TEXT, SAMPLE_TEXT)
    print("\n== Document Sentiment ===")

    for label, content in zip(content_labels, content_text):
        print(f"\nText: {label}")
        blob, vader = content_sentiment(content)
        print(f"Content Sentiment:", blob, vader)
        
    print("\n== Sentence Sentiment ===")
    print(f"\nText: sample_text")
    sentences, bart, vader = sentence_sentiment(SAMPLE_TEXT)
    print(f"Sentence Sentiment:")
    for s, b, v in zip(sentences, bart, vader):
        clean_text = "'" + " ".join(s.strip().split())[:60] + "...':"
        print(f"{clean_text:<64}", b, v)


if __name__ == "__main__":
    demo_sentiment()
