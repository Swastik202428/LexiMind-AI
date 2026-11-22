"""Part-of-speech tagging through spaCy, NLTK, or a transparent heuristic."""

import re

from utils.helpers import validate_text
from utils.nlp_resources import load_spacy_model

PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those"}
PREPOSITIONS = {"in", "on", "at", "by", "for", "from", "to", "with", "of", "about", "under", "over"}
CONJUNCTIONS = {"and", "or", "but", "because", "so", "if", "while"}
AUXILIARIES = {"am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did"}


def _heuristic_tag(word: str) -> str:
    token = word.casefold()
    if token in PRONOUNS:
        return "PRON"
    if token in DETERMINERS:
        return "DET"
    if token in PREPOSITIONS:
        return "ADP"
    if token in CONJUNCTIONS:
        return "CCONJ"
    if token in AUXILIARIES:
        return "AUX"
    if token.endswith("ly"):
        return "ADV"
    if token.endswith(("ous", "ful", "ive", "able", "al", "ic")):
        return "ADJ"
    if token.endswith(("ing", "ed")):
        return "VERB"
    if word[:1].isupper():
        return "PROPN"
    return "NOUN"


def tag_text(text: str) -> dict:
    validate_text(text)
    model, unavailable_reason = load_spacy_model()
    if model is not None and ("tagger" in model.pipe_names or "morphologizer" in model.pipe_names):
        doc = model(text)
        return {
            "tags": [{"token": item.text, "tag": item.pos_, "detail": item.tag_} for item in doc if not item.is_space],
            "method": "spaCy en_core_web_sm",
        }
    tokens = re.findall(r"\b[\w'-]+\b|[^\w\s]", text, flags=re.UNICODE)
    try:
        import nltk

        word_positions = [index for index, token in enumerate(tokens) if re.fullmatch(r"[\w'-]+", token)]
        tagged_words = nltk.pos_tag([tokens[index] for index in word_positions])
        word_tags = {position: tag for position, (_, tag) in zip(word_positions, tagged_words)}
        tags = [
            {"token": token, "tag": word_tags.get(index, "PUNCT"), "detail": word_tags.get(index, "PUNCT")}
            for index, token in enumerate(tokens)
        ]
        return {"tags": tags, "method": "NLTK averaged perceptron tagger"}
    except (ImportError, LookupError):
        return {
            "tags": [
                {
                    "token": token,
                    "tag": _heuristic_tag(token) if re.fullmatch(r"[\w'-]+", token) else "PUNCT",
                    "detail": "heuristic",
                }
                for token in tokens
            ],
            "method": f"rule-based fallback ({unavailable_reason or 'no local POS model available'})",
        }
