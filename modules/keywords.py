"""TF-IDF and frequency-ranked keyword extraction."""

import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

from utils.helpers import STOPWORDS, tokenize_words, validate_text


def extract_keywords(text: str, top_n: int = 10) -> dict:
    validate_text(text)
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("The keyword limit must be a positive integer.")
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    words = [word.casefold() for word in tokenize_words(text) if word.casefold() not in STOPWORDS]
    frequencies = Counter(words)
    tfidf_scores = {}
    method = "frequency fallback (IDF requires multiple usable sentences)"
    if len(sentences) > 1:
        vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS), ngram_range=(1, 2), max_features=500)
        try:
            matrix = vectorizer.fit_transform(sentences)
            names = vectorizer.get_feature_names_out()
            tfidf_scores = dict(zip(names, matrix.sum(axis=0).A1.tolist()))
            method = "sentence-level TF-IDF aggregated across the input"
        except ValueError:
            tfidf_scores = {}
    if not tfidf_scores:
        tfidf_scores = {word: float(count) for word, count in frequencies.items()}
    return {
        "tfidf": sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)[:top_n],
        "frequency": frequencies.most_common(top_n),
        "method": method,
    }
