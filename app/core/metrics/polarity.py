import spacy

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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


# Example usage and testing function
def demo_polarity():
    """Test the polarity functions with different parameters"""

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

    print("\n== Document Polarity ===")

    for label, content in zip(content_labels, (negative_text, neutral_text, positive_text)):
        print(f"\nText: {label}")
        
        blob, vader = content_polarity(content)
        print(f"Content Polarity:", blob, vader)
        
    print("\n== Sentence Polarity ===")
    print(f"\nText: sample_text")
    # Textblob and vader polarity scores range from [-1.0, 1.0] with 1 being the most positive
    sentences, blob, vader = sentence_polarity(sample_text)
    
    # Format document sentence in a more readable way
    print(f"Sentence Polarity:")
    for s, b, v in zip(sentences, blob, vader):
        clean_text = "'" + " ".join(s.strip().split())[:60] + "...':"
        print(f"{clean_text:<64}", b, v)


if __name__ == "__main__":
    demo_polarity()
