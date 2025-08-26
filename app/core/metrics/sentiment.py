import spacy

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from app.core.metrics.categories import classify_content

# Define sentiment class constant
SENTIMENT_CLASSES = ["negative", "neutral", "positive"]

# Define module-level variables
vader_analyzer = SentimentIntensityAnalyzer()
spacy_nlp = spacy.load("en_core_web_lg")


def content_sentiment(content: str) -> dict:
    """Compute bart and vader sentiment scores for the supplied string"""
    doc = spacy_nlp(content)

    # Get bart and vader scores in an equivalent format (including precision)
    bart_scores = classify_content(doc.text, SENTIMENT_CLASSES)
    bart_scores = [round(score, 3) for score in bart_scores]
    vader_scores = vader_analyzer.polarity_scores(doc.text)
    vader_scores = [vader_scores[k] for k in ('neg', 'neu', 'pos')]

    return bart_scores, vader_scores


def sentence_sentiment(content: str) -> tuple[list]:
    """Compute bart and vader sentiment scores for each sentence in the supplied string"""
    doc = spacy_nlp(content)

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


# Example usage and testing function
def demo_sentiment():
    """Test the sentiment functions with different parameters"""

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

    print("\n== Document Sentiment ===")

    for label, content in zip(content_labels, (negative_text, neutral_text, positive_text)):
        print(f"\nText: {label}")
        blob, vader = content_sentiment(content)
        print(f"Content Sentiment:", blob, vader)
        
    print("\n== Sentence Sentiment ===")
    print(f"\nText: sample_text")
    sentences, bart, vader = sentence_sentiment(sample_text)
    print(f"Sentence Sentiment:")
    for s, b, v in zip(sentences, bart, vader):
        clean_text = "'" + " ".join(s.strip().split())[:60] + "...':"
        print(f"{clean_text:<64}", b, v)


if __name__ == "__main__":
    demo_sentiment()
