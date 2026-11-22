import unittest

from modules.keywords import extract_keywords


class KeywordTests(unittest.TestCase):
    def test_returns_ranked_frequency_and_tfidf_keywords(self):
        result = extract_keywords(
            "Language models analyze text. Language tools help analyze documents.",
            top_n=5,
        )
        self.assertEqual(result["frequency"][0], ("language", 2))
        self.assertLessEqual(len(result["tfidf"]), 5)
        self.assertTrue(all(score >= 0 for _, score in result["tfidf"]))

    def test_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            extract_keywords("")

    def test_identifies_frequency_fallback_when_idf_is_unavailable(self):
        result = extract_keywords("Language models process language.")
        self.assertIn("frequency fallback", result["method"])


if __name__ == "__main__":
    unittest.main()
