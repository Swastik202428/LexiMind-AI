"""Extractive sentence ranking using normalized word frequencies."""

import re
from collections import Counter

from utils.helpers import STOPWORDS, tokenize_words, validate_text


def summarize_text(text: str, sentence_count: int = 3) -> dict:
    validate_text(text)
    if not isinstance(sentence_count, int) or isinstance(sentence_count, bool) or sentence_count < 1:
        raise ValueError("The summary sentence count must be a positive integer.")
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    if not sentences:
        return {"summary": "", "sentences": [], "method": "word-frequency extractive ranking"}
    words = [word.casefold() for word in tokenize_words(text) if word.casefold() not in STOPWORDS]
    frequencies = Counter(words)
    max_frequency = max(frequencies.values(), default=1)
    scores = []
    for index, sentence in enumerate(sentences):
        sentence_words = [word.casefold() for word in tokenize_words(sentence)]
        score = sum(frequencies[word] / max_frequency for word in sentence_words if word in frequencies)
        scores.append((score / max(len(sentence_words), 1), index, sentence))
    count = min(sentence_count, len(sentences))
    selected = sorted(sorted(scores, key=lambda item: (-item[0], item[1]))[:count], key=lambda item: item[1])
    summary_sentences = [item[2] for item in selected]
    return {
        "summary": " ".join(summary_sentences),
        "sentences": summary_sentences,
        "method": "extractive word-frequency sentence ranking",
    }
