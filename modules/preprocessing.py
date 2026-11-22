"""Text normalization with optional NLTK and spaCy enhancements."""

import re
import string
import unicodedata

from utils.helpers import (
    detect_text_language,
    remove_unicode_numbers,
    remove_unicode_punctuation,
    stopword_resource_name,
    stopwords_for_text,
    tokenize_words,
    validate_text,
)
from utils.nlp_resources import is_wordnet_available


def _fallback_lemma(word: str) -> str:
    """Apply conservative English suffix rules when WordNet is unavailable."""
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            stem = word[: -len(suffix)]
            if suffix == "ies":
                return stem + "y"
            if suffix in {"ing", "ed"} and len(stem) > 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
    return word


def _hindi_suffix_stem(word: str) -> str:
    """Apply conservative Hindi suffix rules when a language-specific stemmer is unavailable."""
    if word == "रही":
        return "रह"
    suffixes = (
        "रहे", "रहा", "ओंगी", "ओगे", "ओगी", "ाएगी", "एगी", "ाएगा", "ेगा",
    )
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 1:
            return word[: -len(suffix)]
    return word


def _is_devanagari_word(word: str) -> bool:
    return bool(re.search(r"[क-ह]", word))


def _remove_unicode_special_characters(text: str) -> str:
    return "".join(
        " " if unicodedata.category(character).startswith(("P", "S")) else character
        for character in text
    )


def process_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_special_characters: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    stemming: bool = False,
    lemmatization: bool = False,
) -> dict:
    """Return original text, transformed text, and the operations applied."""
    validate_text(text)

    result = text
    applied = []
    resource_warning = None
    if lowercase:
        result = result.lower()
        applied.append("lowercase")
    if remove_numbers:
        result = remove_unicode_numbers(result)
        applied.append("numbers removed")
    if remove_punctuation:
        result = remove_unicode_punctuation(result)
        result = result.translate(str.maketrans({character: " " for character in string.punctuation}))
        applied.append("punctuation removed")
    if remove_special_characters:
        result = _remove_unicode_special_characters(result)
        applied.append("special characters removed")

    words = tokenize_words(result)
    if remove_stopwords:
        stopwords = stopwords_for_text(text)
        words = [word for word in words if word.casefold() not in stopwords]
        applied.append(f"stopwords removed ({stopword_resource_name(text)})")
    if stemming:
        language = detect_text_language(text)
        if language in {"hi", "mixed"}:
            words = [
                _hindi_suffix_stem(word) if _is_devanagari_word(word) else word
                for word in words
            ]
            applied.append("rule-based Hindi suffix stemming")
            if language == "hi":
                resource_warning = "No dedicated Hindi stemmer is configured. Using conservative built-in Hindi suffix rules."
            else:
                resource_warning = "Mixed text uses a conservative Hindi suffix fallback for Devanagari words."
        if language in {"en", "mixed"}:
            try:
                from nltk.stem import PorterStemmer

                stemmer = PorterStemmer()
                words = [
                    stemmer.stem(word) if not _is_devanagari_word(word) else word
                    for word in words
                ]
                applied.append("Porter stemming")
            except ImportError:
                raise RuntimeError("NLTK is required for English stemming. Install it from requirements.txt.")
    if lemmatization:
        language = detect_text_language(text)
        if language in {"hi", "mixed"}:
            words = [
                _hindi_suffix_stem(word) if _is_devanagari_word(word) else word
                for word in words
            ]
            applied.append("rule-based Hindi suffix lemmatization")
            resource_warning = "WordNet is English-only. Devanagari words use conservative built-in Hindi suffix rules."

        if language in {"en", "mixed"}:
            wordnet_available, wordnet_reason = is_wordnet_available()
            if wordnet_available:
                try:
                    from nltk.stem import WordNetLemmatizer

                    lemmatizer = WordNetLemmatizer()
                    words = [
                        lemmatizer.lemmatize(word) if not _is_devanagari_word(word) else word
                        for word in words
                    ]
                    applied.append("WordNet lemmatization")
                except (ImportError, LookupError, OSError):
                    words = [
                        _fallback_lemma(word) if not _is_devanagari_word(word) else word
                        for word in words
                    ]
                    applied.append("rule-based English lemmatization (WordNet data unavailable)")
                    resource_warning = f"{wordnet_reason or 'WordNet data is unavailable.'} Using an English suffix-based fallback."
            else:
                words = [
                    _fallback_lemma(word) if not _is_devanagari_word(word) else word
                    for word in words
                ]
                applied.append("rule-based English lemmatization")
                resource_warning = f"{wordnet_reason or 'WordNet data is unavailable.'} Using an English suffix-based fallback."

    return {
        "original": text,
        "cleaned": " ".join(words),
        "operations": applied,
        "word_count_before": len(tokenize_words(text)),
        "word_count_after": len(words),
        "resource_warning": resource_warning,
    }
