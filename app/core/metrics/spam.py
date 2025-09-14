import numpy
import torch

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.core.utils.samples import SPAM_TEXT, HAM_TEXT, NEGATIVE_TEXT, NEUTRAL_TEXT, POSITIVE_TEXT, SAMPLE_TEXT


# Get pre-trained toxicity and spam detection models
tokenizer = AutoTokenizer.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
spam_classifier = AutoModelForSequenceClassification.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
toxic_classifier = pipeline("text-classification", model="unitary/toxic-bert")


def score_spam(content: str) -> float:
    """Compute spam scores for the supplied text content"""
    # Tokenize the input
    inputs = tokenizer(content, return_tensors="pt")
    
    # Get model predictions
    with torch.no_grad():
        outputs = spam_classifier(**inputs)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1)

    return round(float(probabilities.flatten()[1]), 4)


def score_toxicity(content: str) -> float:
    """Compute toxicity scores for the supplied text content"""
    # Simply apply the toxicity classifier to the input
    return round(toxic_classifier(content)[0]['score'], 4)


# Example usage and testing function
def demo_spam():
    """Test the spam scoring functions with different parameters"""
    # Group all labels and content
    content_labels = ('spam', 'ham', 'negative', 'neutral', 'positive', 'document')
    content_strings = (SPAM_TEXT, HAM_TEXT, NEGATIVE_TEXT, NEUTRAL_TEXT, POSITIVE_TEXT, SAMPLE_TEXT)

    print("\n== Spam Classification ===")
    for label, content in zip(content_labels, content_strings):
        # Get predicted labels and map labels to class names
        spam_score = score_spam(content)
        label = "Not Spam" if spam_score < 0.5 else "Spam"
        prediction = int(numpy.argmax(spam_score))
        print(f"\n{label.capitalize()} text: {content}\nPrediction: {label}")
        
        print("Spam score:", spam_score)
        
        toxicity_score = score_toxicity(content)
        print("Toxicity score:", toxicity_score)


if __name__ == "__main__":
    demo_spam()
