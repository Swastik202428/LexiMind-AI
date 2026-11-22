"""A small deterministic FAQ chatbot for NLP concepts and common intents."""

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.helpers import validate_text

FAQS = [
    ("what is nlp natural language processing", "Natural Language Processing (NLP) is the area of AI that helps computers analyze, understand, and work with human language."),
    ("what is tokenization", "Tokenization splits text into smaller units such as sentences, words, or subwords."),
    ("what are stop words", "Stop words are very common words, such as 'the' and 'is', that are often removed when they add little value to a particular text analysis."),
    ("what is sentiment analysis", "Sentiment analysis estimates the polarity or opinion expressed in text. Lexicon-based methods are useful baselines, not perfect measures of intent."),
    ("what is named entity recognition ner", "Named Entity Recognition (NER) finds spans such as people, organizations, locations, dates, and monetary amounts."),
    ("what is tf idf", "TF-IDF weights terms by how frequent they are in a document and how uncommon they are across a document collection."),
    ("what is a language model", "A language model learns statistical patterns in language. This demo chatbot is rule/FAQ based and is not a generative language model."),
    ("what is topic modeling lda", "Topic modeling groups recurring terms across documents. LDA is a probabilistic method that represents documents as mixtures of topics."),
    ("what is pos tagging", "Part-of-speech tagging labels words by grammatical role, such as noun, verb, adjective, or pronoun."),
    ("what is stemming lemmatization", "Stemming chops word endings using simple rules; lemmatization maps a word to a dictionary form using linguistic knowledge."),
    ("help what can you do", "Ask about NLP, tokenization, sentiment, TF-IDF, NER, POS tagging, or topic modeling. Try: 'What is tokenization?'"),
]


@lru_cache(maxsize=1)
def _faq_index():
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([question for question, _ in FAQS])
    return vectorizer, matrix


def respond(message: str) -> dict:
    validate_text(message, "Message")
    text = message.strip()
    normalized = text.casefold().strip(" !.,?")
    if normalized in {"hi", "hello", "hey", "good morning", "good afternoon"}:
        return {"intent": "greeting", "reply": "Hello! I can answer basic questions about NLP concepts. Type “help” to see examples."}
    if normalized in {"help", "what can you do", "commands"}:
        return {"intent": "help", "reply": FAQS[-1][1]}
    if normalized in {"bye", "goodbye", "see you"}:
        return {"intent": "goodbye", "reply": "Goodbye! Your conversation stays in this browser session."}
    vectorizer, faq_matrix = _faq_index()
    query = vectorizer.transform([text])
    scores = cosine_similarity(query, faq_matrix).ravel()
    best = int(scores.argmax())
    if float(scores[best]) < 0.12:
        reply = "I’m a lightweight NLP concept bot, not an LLM. I don’t know that one yet—try asking about tokenization, sentiment analysis, TF-IDF, NER, POS tagging, or topic modeling."
        return {"intent": "fallback", "reply": reply, "match_score": float(scores[best])}
    return {"intent": "nlp_faq", "reply": FAQS[best][1], "match_score": float(scores[best])}
