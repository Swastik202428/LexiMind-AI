import unittest

from modules.embeddings import vectorize_text, word_similarity


class EmbeddingTests(unittest.TestCase):
    def test_bow_and_tfidf_shapes_match_documents_and_features(self):
        result = vectorize_text("Language models process language.\nModels learn from documents.")
        self.assertEqual(len(result["bow"]), 2)
        self.assertEqual(len(result["tfidf"]), 2)
        self.assertEqual(len(result["bow"][0]), len(result["bow_features"]))
        self.assertEqual(len(result["tfidf"][1]), len(result["tfidf_features"]))

    def test_context_similarity_is_bounded(self):
        text = "language tools process text. language systems process documents."
        result = word_similarity(text, "language", "tools")
        self.assertGreaterEqual(result["similarity"], 0)
        self.assertLessEqual(result["similarity"], 1)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            vectorize_text("")

    def test_rejects_invalid_context_window(self):
        with self.assertRaises(ValueError):
            word_similarity("some text here", "some", "text", window=0)


if __name__ == "__main__":
    unittest.main()
