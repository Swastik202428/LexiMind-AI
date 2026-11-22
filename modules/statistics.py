"""Descriptive text statistics and ranked term frequencies."""

import re
from collections import Counter

from utils.helpers import split_sentences, tokenize_words, validate_text


def analyze_text(text: str, limit: int = 12) -> dict:
    validate_text(text)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("The frequency limit must be a positive integer.")
    words = tokenize_words(text)
    sentences = split_sentences(text)
    frequencies = Counter(word.casefold() for word in words)
    return {
        "word_count": len(words),
        "character_count": len(text),
        "character_count_no_spaces": len(re.sub(r"\s", "", text)),
        "sentence_count": len(sentences),
        "average_word_length": sum(map(len, words)) / len(words) if words else 0.0,
        "vocabulary_size": len(frequencies),
        "frequencies": frequencies.most_common(limit),
    }
