import spacy

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.core.utils.samples import NEGATIVE_TEXT, NEUTRAL_TEXT, POSITIVE_TEXT, SAMPLE_TEXT


# Define module-level variables
vader_analyzer = SentimentIntensityAnalyzer()
spacy_nlp = spacy.load("en_core_web_lg")


def content_polarity(content: str) -> dict:
    """Compute blob and vader polarity for the supplied string"""
    doc = spacy_nlp(content)

    # For both sets of scores: -1 most extreme negative, +1 most extreme positive
    blob_score = TextBlob(doc.text).sentiment.polarity
    vader_score = vader_analyzer.polarity_scores(doc.text)['compound']

    return round(blob_score, 4), round(vader_score, 4)


def sentence_polarity(content: str) -> list:
    """Compute blob and vader polarity for each sentence in the supplied string"""
    doc = spacy_nlp(content)

    sentence_list, blob_list, vader_list = [], [], []
    for sentence in doc.sents:
        # For both sets of scores: -1 most extreme negative, +1 most extreme positive
        sentence_list.append(sentence.text)

        blob_score = TextBlob(sentence.text).sentiment.polarity
        blob_list.append(round(blob_score, 4))

        vader_score = vader_analyzer.polarity_scores(sentence.text)['compound']
        vader_list.append(round(vader_score, 4))

    return sentence_list, blob_list, vader_list


def score_polarity(content: str) -> float:
    """Return the composite (mean) polarity score for the supplied content"""
    blob_score, vader_score = content_polarity(content)
    
    return (blob_score + vader_score) / 2


# Example usage and testing function
def demo_polarity():
    """Test the polarity functions with different parameters"""
    content_labels = ('negative', 'neutral', 'positive', 'document')
    content_text = (NEGATIVE_TEXT, NEUTRAL_TEXT, POSITIVE_TEXT, SAMPLE_TEXT)

    print("\n== Document Polarity ===")
    for label, content in zip(content_labels, content_text):
        print(f"\nText: {label}")
        
        blob_score, vader_score = content_polarity(content)
        print(f"Content Polarity:", blob_score, vader_score)
        
    print("\n== Sentence Polarity ===")
    print(f"\nText: sample_text")
    # Textblob and vader_score polarity scores range from [-1.0, 1.0] with 1 being the most positive
    sentences, blob_score, vader_score = sentence_polarity(SAMPLE_TEXT)
    
    # Format document sentence in a more readable way
    print(f"Sentence Polarity:")
    for s, b, v in zip(sentences, blob_score, vader_score):
        clean_text = "'" + " ".join(s.strip().split())[:60] + "...':"
        print(f"{clean_text:<64}", b, v)


if __name__ == "__main__":
    demo_polarity()
