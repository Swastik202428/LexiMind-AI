"""Shared text helpers, session metrics, and downloadable reports."""

import re
import unicodedata
from datetime import datetime

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "ours",
    "she", "so", "than", "that", "the", "their", "theirs", "them", "then", "there", "these",
    "they", "this", "those", "to", "too", "up", "us", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your", "yours",
}

HINDI_STOPWORDS = {
    "अतः", "और", "अगर", "एक", "इस", "इसका", "इसके", "इसे", "उसे", "उसका",
    "उसके", "उससे", "क्या", "की", "के", "को", "कौन", "था", "थे", "थी",
    "न", "नहीं", "पर", "पहले", "फिर", "बाद", "बिना", "भी", "मैं", "मे", "मेरा",
    "यह", "वह", "वे", "है", "हैं", "हो", "होता", "होती", "होते", "हुआ",
    "हुए", "हुई", "रही", "रहे", "रहा", "से", "ही", "तक",
}


def validate_text(text: str, field_name: str = "Text") -> str:
    """Require a non-empty string at public NLP service boundaries."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def tokenize_words(text: str) -> list:
    """Return English/Latin and Devanagari words without splitting Unicode marks."""
    pattern = r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[\u0900-\u0963\u0971-\u097F]+"
    return re.findall(pattern, text, flags=re.UNICODE)


def split_sentences(text: str) -> list:
    """Split on English, Unicode, or Devanagari sentence terminators."""
    parts = re.split(
        r"(?<=[!?。！？।॥])(?:\s+|(?=[^\s]))|(?<=\.)(?:\s+|(?=[A-Za-z\u0900-\u097F]))",
        text.strip(),
    )
    return [part.strip() for part in parts if part.strip()]


def detect_text_language(text: str) -> str:
    has_latin = bool(re.search(r"[A-Za-z]", text))
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    if has_latin and has_devanagari:
        return "mixed"
    if has_devanagari:
        return "hi"
    if has_latin:
        return "en"
    return "unknown"


def stopwords_for_text(text: str) -> set:
    language = detect_text_language(text)
    if language == "hi":
        return set(HINDI_STOPWORDS)
    if language == "mixed":
        return set(STOPWORDS) | set(HINDI_STOPWORDS)
    return set(STOPWORDS)


def stopword_resource_name(text: str) -> str:
    language = detect_text_language(text)
    if language == "hi":
        return "Hindi built-in stopwords"
    if language == "mixed":
        return "English and Hindi built-in stopwords"
    return "English built-in stopwords"


def remove_unicode_punctuation(text: str) -> str:
    return "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    )


def remove_unicode_numbers(text: str) -> str:
    return "".join(
        " " if character.isdigit() or unicodedata.category(character) in {"Nd", "Nl", "No"} else character
        for character in text
    )


def initialize_session_state(state) -> None:
    state.setdefault("total_words_processed", 0)
    state.setdefault("analysis_count", 0)
    state.setdefault("analysis_history", [])
    state.setdefault("chat_history", [])


def record_analysis(state, module: str, text: str, result: str = "") -> None:
    initialize_session_state(state)
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "module": module,
        "words": len(tokenize_words(text)),
        "preview": text.strip().replace("\n", " ")[:120],
        "result": str(result)[:500],
    }
    state["analysis_history"].insert(0, entry)
    state["analysis_history"] = state["analysis_history"][:25]
    state["total_words_processed"] += entry["words"]
    state["analysis_count"] += 1


def history_csv(history: list) -> str:
    import csv
    from io import StringIO

    buffer = StringIO()
    columns = ["time", "module", "words", "preview", "result"]
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(history)
    return buffer.getvalue()


def history_report(history: list) -> str:
    if not history:
        return "LexiMind AI analysis report\n\nNo analyses have been recorded in this session."
    lines = ["LexiMind AI — Session Analysis Report", "=" * 40, ""]
    for item in history:
        lines.extend([
            f"[{item['time']}] {item['module']} — {item['words']} words",
            f"Input: {item['preview']}",
            f"Result: {item['result']}",
            "",
        ])
    return "\n".join(lines)
