"""Extractive question answering using sentence-level TF-IDF similarity."""

import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.helpers import validate_text

MIN_RELEVANCE_SCORE = 0.1
QA_STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "does", "did", "do", "what", "how", "which", "where", "when", "why", "who",
}
STEMMER = PorterStemmer()


def _normalized_tokens(text: str) -> list[str]:
    """Stem content terms so simple inflections can match in retrieval."""
    words = re.findall(r"\b[\w'-]+\b", text.casefold(), flags=re.UNICODE)
    return [STEMMER.stem(word) for word in words if word not in QA_STOPWORDS]


def answer_question(context: str, question: str) -> dict:
    validate_text(context, "Context")
    validate_text(question, "Question")
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", context.strip()) if item.strip()]
    if not sentences:
        raise ValueError("The context does not contain an answerable sentence.")
    try:
        vectorizer = TfidfVectorizer(
            tokenizer=_normalized_tokens,
            token_pattern=None,
            ngram_range=(1, 2),
        )
        matrix = vectorizer.fit_transform(sentences + [question.strip()])
    except ValueError as error:
        raise ValueError("Add more descriptive words to the context and question.") from error
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    best = int(scores.argmax())
    relevance = float(scores[best])
    supported = relevance >= MIN_RELEVANCE_SCORE
    return {
        "answer": sentences[best] if supported else None,
        "relevance": relevance,
        "supported": supported,
        "threshold": MIN_RELEVANCE_SCORE,
        "method": "extractive sentence retrieval using TF-IDF cosine similarity; not a generative LLM",
    }
