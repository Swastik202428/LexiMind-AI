import unittest

from modules.topic_modeling import model_topics


class TopicModelingTests(unittest.TestCase):
    def test_lda_returns_requested_number_of_ranked_topics(self):
        documents = [
            "Researchers study language data and text patterns.",
            "Businesses report revenue, sales, and market growth.",
            "Scientists conduct laboratory research and publish findings.",
        ]
        result = model_topics(documents, topic_count=2, words_per_topic=4)
        self.assertEqual(len(result["topics"]), 2)
        self.assertEqual(result["document_count"], 3)
        self.assertTrue(all(len(topic["words"]) == 4 for topic in result["topics"]))

    def test_requires_multiple_documents(self):
        with self.assertRaises(ValueError):
            model_topics(["Only one document"])

    def test_rejects_non_positive_topic_count(self):
        with self.assertRaises(ValueError):
            model_topics(["language and text processing", "text and language research"], topic_count=0)


if __name__ == "__main__":
    unittest.main()
