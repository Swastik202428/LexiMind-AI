"""Offline language detection via langdetect with short-input safeguards."""

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

from utils.helpers import validate_text

DetectorFactory.seed = 0

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh-cn": "Chinese (Simplified)",
    "ja": "Japanese",
    "ar": "Arabic",
    "bn": "Bengali",
    "ur": "Urdu",
}


def detect_language(text: str) -> dict:
    validate_text(text)
    value = text.strip()
    if len(value) < 4:
        if any("\u0900" <= char <= "\u097f" for char in value):
            return {"code": "hi", "name": "Hindi", "confidence": None, "note": "Detected from Devanagari script; input is very short."}
        return {"code": None, "name": "Too short to identify reliably", "confidence": None, "note": "Add a few words for a reliable result."}
    try:
        candidates = detect_langs(value)
    except LangDetectException as error:
        raise ValueError("Language detection could not identify this text. Try a longer sample.") from error
    if not candidates:
        raise ValueError("Language detection returned no candidates. Try a longer sample.")
    code = candidates[0].lang
    return {
        "code": code,
        "name": LANGUAGE_NAMES.get(code, code.upper()),
        "confidence": float(candidates[0].prob) if candidates else None,
        "candidates": [
            {"code": candidate.lang, "confidence": float(candidate.prob)}
            for candidate in candidates[:5]
        ],
        "note": "Language detection is approximate, especially for short or mixed-language text.",
    }
