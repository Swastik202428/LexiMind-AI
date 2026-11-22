"""Bag-of-Words, TF-IDF, and lightweight context-vector similarities."""

from collections import Counter, defaultdict

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.helpers import tokenize_words, validate_text


def vectorize_text(text: str, max_features: int = 30) -> dict:
    validate_text(text)
    if not isinstance(max_features, int) or isinstance(max_features, bool) or max_features < 1:
        raise ValueError("The maximum feature count must be a positive integer.")
    documents = [sentence.strip() for sentence in text.splitlines() if sentence.strip()]
    if not documents:
        documents = [text]
    count = CountVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    tfidf = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    try:
        bow_matrix = count.fit_transform(documents)
        tfidf_matrix = tfidf.fit_transform(documents)
    except ValueError as error:
        raise ValueError("Add at least one word with letters or numbers to vectorize.") from error
    return {
        "documents": documents,
        "bow_features": count.get_feature_names_out().tolist(),
        "bow": bow_matrix.toarray().tolist(),
        "tfidf_features": tfidf.get_feature_names_out().tolist(),
        "tfidf": tfidf_matrix.toarray().round(4).tolist(),
    }


def word_similarity(text: str, first: str, second: str, window: int = 3) -> dict:
    validate_text(text)
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("The context window must be a positive integer.")
    if not isinstance(first, str) or not isinstance(second, str):
        raise ValueError("Both comparison words must be strings.")
    tokens = [token.casefold() for token in tokenize_words(text)]
    first, second = first.casefold().strip(), second.casefold().strip()
    if not first or not second:
        raise ValueError("Enter both words to compare.")
    contexts = defaultdict(Counter)
    for index, token in enumerate(tokens):
        for offset in range(max(0, index - window), min(len(tokens), index + window + 1)):
            if offset != index:
                contexts[token][tokens[offset]] += 1
    if first not in contexts or second not in contexts:
        return {"similarity": 0.0, "method": "local context co-occurrence; one or both words were not found"}
    features = sorted(set(contexts[first]) | set(contexts[second]))
    vectors = [[contexts[word][feature] for feature in features] for word in (first, second)]
    score = float(cosine_similarity(vectors)[0, 1]) if any(vectors[0]) and any(vectors[1]) else 0.0
    return {"similarity": score, "method": f"cosine similarity of context counts (window={window})"}
