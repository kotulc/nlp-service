from app.core.metrics import polarity, sentiment, spam, style
from app.core.summary import headings, generate
from app.core.tags import extract


def test_polarity():
    """Test the polarity metric functions"""
    polarity.demo_polarity()


def test_sentiment():
    """Test the sentiment metric functions"""
    sentiment.demo_sentiment()


def test_spam():
    """Test the spam metric functions"""
    spam.demo_spam()


def test_style():
    """Test the toxicity metric functions"""
    style.demo_style()


def test_generator():
    """Test the summary generation functions"""
    generate.demo_generator()


def test_headings():
    """Test the heading generation functions"""
    headings.demo_headings()


def test_extract():
    """Test the tagging functions"""
    extract.demo_tagger()
