"""Lexicon-based sentiment analysis using TextBlob."""

from textblob import TextBlob

from utils.helpers import validate_text


def analyze_sentiment(text: str) -> dict:
    validate_text(text)
    scores = TextBlob(text).sentiment
    polarity = float(scores.polarity)
    label = "Positive" if polarity > 0.05 else "Negative" if polarity < -0.05 else "Neutral"
    return {
        "label": label,
        "polarity": polarity,
        "subjectivity": float(scores.subjectivity),
        "method": "TextBlob lexicon-based polarity",
    }
