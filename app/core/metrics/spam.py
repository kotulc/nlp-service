import numpy
import torch

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Get pre-trained toxicity and spam detection models
tokenizer = AutoTokenizer.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
spam_classifier = AutoModelForSequenceClassification.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
toxic_classifier = pipeline("text-classification", model="unitary/toxic-bert")


def score_spam(content: str) -> list:
    """Compute spam scores for the supplied text content"""
    # Tokenize the input
    inputs = tokenizer(content, return_tensors="pt")
    
    # Get model predictions
    with torch.no_grad():
        outputs = spam_classifier(**inputs)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1)    
    return [round(float(score), 4) for score in probabilities.flatten()]


def score_toxicity(content: str) -> list:
    """Compute toxicity scores for the supplied text content"""
    # Simply apply the toxicity classifier to the input
    return round(toxic_classifier(content)[0]['score'], 4)


# Example usage and testing function
def demo_spam():
    """Test the spam scoring functions with different parameters"""

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

    # Generic text
    negative_text = "I hate AI, it is the worst thing to happen to humanity ever"
    neutral_text = "AI may be a devistating force, it could possibly mean the end of the world"
    positive_text = "AI will bring about a utopian revolution, it will be very beneficial"

    # Sample spam/ham text
    spam_text = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim now."
    ham_text = "Hey, are we still meeting for lunch today?"

    # Group all labels and content
    content_labels = ('spam', 'ham', 'negative', 'neutral', 'positive', 'document')
    content_strings = (spam_text, ham_text, negative_text, neutral_text, positive_text, sample_text)

    print("\n== Spam Classification ===")
    for label, content in zip(content_labels, content_strings):
        # Get predicted labels and map labels to class names
        label_map = {0: "Not Spam", 1: "Spam"}
        spam_scores = score_spam(content)
        prediction = int(numpy.argmax(spam_scores))
        print(f"\n{label.capitalize()} text: {content}\nPrediction: {label_map[prediction]}")
        
        print("Spam scores:", spam_scores)
        
        toxicity_score = score_toxicity(content)
        print("Toxicity scores:", toxicity_score)


if __name__ == "__main__":
    demo_spam()
