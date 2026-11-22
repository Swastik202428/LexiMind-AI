"""Sentence, word, and n-gram tokenization."""

import re
from collections import Counter

from utils.helpers import split_sentences, tokenize_words, validate_text


def tokenize(text: str) -> dict:
    validate_text(text)
    sentences = split_sentences(text)
    words = tokenize_words(text)
    return {"sentences": sentences, "words": words, "word_count": len(words)}


def ngrams(text: str, n: int) -> list:
    validate_text(text)
    if n not in (1, 2, 3):
        raise ValueError("Only unigrams, bigrams, and trigrams are supported.")
    words = [word.casefold() for word in tokenize_words(text)]
    return [" ".join(words[index : index + n]) for index in range(len(words) - n + 1)]


def ngram_counts(text: str) -> dict:
    return {f"{n}-grams": Counter(ngrams(text, n)) for n in (1, 2, 3)}
