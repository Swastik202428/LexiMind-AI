"""Named-entity recognition with an optional spaCy model and local regex fallback."""

import re

from utils.helpers import validate_text
from utils.nlp_resources import load_spacy_model


def _regex_entities(text: str) -> list:
    found = []
    patterns = {
        "DATE": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?|\d{4})\b",
        "MONEY": r"(?:[$€£₹]\s?\d[\d,]*(?:\.\d{1,2})?|\b\d[\d,]*(?:\.\d{1,2})?\s?(?:USD|INR|dollars?|rupees?|euros?)\b)",
        "ORG": r"\b(?:Google|Microsoft|OpenAI|Apple|Amazon|Meta|NASA|UN|WHO|RGPV|IIT|University)\b(?:\s+[A-Z][a-zA-Z&.-]+){0,3}",
        "LOCATION": r"\b(?:India|United States|USA|United Kingdom|UK|London|Paris|Tokyo|Delhi|New York|Mumbai|Bhopal|California|Europe|Asia)\b",
    }
    for label, pattern in patterns.items():
        flags = 0 if label == "ORG" else re.IGNORECASE
        found.extend(
            {"text": match.group(0), "label": label, "start": match.start(), "end": match.end()}
            for match in re.finditer(pattern, text, flags=flags)
        )
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", text):
        candidate = match.group(0)
        if not any(item["start"] < match.end() and item["end"] > match.start() for item in found):
            found.append({"text": candidate, "label": "PERSON", "start": match.start(), "end": match.end()})
    found.sort(key=lambda item: item["start"])
    return found


def extract_entities(text: str) -> dict:
    validate_text(text)
    nlp, unavailable_reason = load_spacy_model()
    if nlp is not None and "ner" in nlp.pipe_names:
        doc = nlp(text)
        return {
            "entities": [
                {"text": entity.text, "label": entity.label_, "start": entity.start_char, "end": entity.end_char}
                for entity in doc.ents
            ],
            "method": "spaCy en_core_web_sm",
            "fallback": False,
        }
    return {
        "entities": _regex_entities(text),
        "method": f"local pattern-based fallback ({unavailable_reason or 'spaCy model has no NER component'})",
        "fallback": True,
    }
