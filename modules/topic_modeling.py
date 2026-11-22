"""Latent Dirichlet Allocation topic extraction from a document collection."""

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def model_topics(documents: list, topic_count: int = 3, words_per_topic: int = 8) -> dict:
    if not isinstance(documents, (list, tuple)) or any(not isinstance(document, str) for document in documents):
        raise ValueError("Documents must be a list of text strings.")
    if not isinstance(topic_count, int) or isinstance(topic_count, bool) or topic_count < 1:
        raise ValueError("The topic count must be a positive integer.")
    if not isinstance(words_per_topic, int) or isinstance(words_per_topic, bool) or words_per_topic < 1:
        raise ValueError("The words-per-topic limit must be a positive integer.")
    clean_documents = [str(document).strip() for document in documents if str(document).strip()]
    if len(clean_documents) < 2:
        raise ValueError("Provide at least two non-empty documents, one per line.")
    vectorizer = CountVectorizer(stop_words="english", max_features=1500, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    try:
        matrix = vectorizer.fit_transform(clean_documents)
    except ValueError as error:
        raise ValueError("The documents need more varied English words to discover topics.") from error
    if matrix.shape[1] < 2:
        raise ValueError("Add more varied words across your documents to discover topics.")
    count = min(topic_count, matrix.shape[1], len(clean_documents))
    model = LatentDirichletAllocation(n_components=count, random_state=42, learning_method="batch", max_iter=10)
    model.fit(matrix)
    vocabulary = vectorizer.get_feature_names_out()
    topics = []
    for index, weights in enumerate(model.components_):
        chosen = weights.argsort()[::-1][:words_per_topic]
        topics.append({
            "topic": index + 1,
            "words": [(str(vocabulary[word]), float(weights[word])) for word in chosen],
        })
    return {
        "topics": topics,
        "document_count": len(clean_documents),
        "feature_count": matrix.shape[1],
        "requested_topic_count": topic_count,
        "topic_count": count,
    }
