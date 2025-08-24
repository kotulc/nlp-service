import spacy

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Define module-level variables
vader_analyzer = SentimentIntensityAnalyzer()
sentiment_classifier = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english')
zero_shot_pipe = pipeline(model='facebook/bart-large-mnli')


# Example usage and testing function
def demo_analyzer():
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

    print("\n== Basic Polarity ===")
    # "compund" score represents the normalized overall sentiment
    # -1 most extreme negative, +1 most extreme positive
    negative_text = "AI will be a devistating foce, it will mean the end of the world when we let it"
    sentiment_classes = ["negative", "neutral", "positive"]
    result = zero_shot_pipe(negative_text, candidate_labels=sentiment_classes)['scores']
    print("test neg. zero-shot:", result)
    
    scores = vader_analyzer.polarity_scores(negative_text)
    print("test neg. vader:", scores)

    positive_text = "AI will bring about a utopian revolution when we use it wisely and guide it"
    result = zero_shot_pipe(positive_text, candidate_labels=sentiment_classes)['scores']
    print("test pos. zero-shot:", result)

    scores = vader_analyzer.polarity_scores(positive_text)
    print("test pos. vader:", scores)

    scores = vader_analyzer.polarity_scores(sample_text)
    print("document vader:", scores)

    result = zero_shot_pipe(sample_text, candidate_labels=sentiment_classes)['scores']
    print("document zero-shot:", result)


    print("\n== Sentence Polarity ===")
    spacy_nlp = spacy.load("en_core_web_lg")
    doc = spacy_nlp(sample_text)
    for sentence in doc.sents:
        clean_tokens = [token.text for token in sentence if not token.is_punct and not token.is_stop and token.is_alpha]
        sentence = " ".join(clean_tokens)
        result = zero_shot_pipe(sentence, candidate_labels=sentiment_classes)['scores']
        print(f"'{sentence[:32]}...':", vader_analyzer.polarity_scores(sentence), result)
    

    print("\n== Document Sentiment ===")
    sentiment_classes = ["dogmatic", "subjective", "balanced", "objective", "grounded"]
    result = zero_shot_pipe(sample_text, candidate_labels=sentiment_classes)['scores']
    print("document objectivity:", result)

    sentiment_classes = ["simplistic", "generic", "concise", "detailed", "nuanced"]
    result = zero_shot_pipe(sample_text, candidate_labels=sentiment_classes)['scores']
    print("document granularity:", result)

    from textblob import TextBlob
    text = TextBlob(sample_text)
    print("textblob subjectivity:", text.sentiment.subjectivity)

    text = TextBlob("AI is the worst. I am completely right about this, nothing can change my mind.")
    print("textblob test:", text.sentiment.subjectivity)


if __name__ == "__main__":
    demo_analyzer()
