"""Lazy, cached access to optional local NLP models."""

from functools import lru_cache


@lru_cache(maxsize=3)
def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Load a spaCy pipeline once, returning a clear reason when unavailable."""
    try:
        import spacy
    except (ImportError, ModuleNotFoundError):
        return None, "The spaCy package is not installed."

    try:
        return spacy.load(model_name), None
    except OSError:
        return None, f"The spaCy model '{model_name}' is not installed."


@lru_cache(maxsize=1)
def is_wordnet_available() -> tuple[bool, str | None]:
    """Check for the local WordNet corpus without downloading or installing it."""
    try:
        import nltk
        from nltk.corpus import wordnet
    except (ImportError, ModuleNotFoundError):
        return False, "The NLTK package is not installed."

    try:
        nltk.data.find("corpora/wordnet")
        wordnet.ensure_loaded()
        return True, None
    except (LookupError, OSError):
        return False, "The WordNet corpus is not installed."
