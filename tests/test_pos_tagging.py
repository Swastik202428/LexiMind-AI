import unittest
from unittest.mock import patch

from modules.pos_tagging import tag_text


class PosTaggingTests(unittest.TestCase):
    def test_tags_words_and_reports_the_method(self):
        result = tag_text("The curious student reads carefully.")
        tokens = [item["token"] for item in result["tags"]]
        self.assertIn("student", tokens)
        self.assertIn("reads", tokens)
        self.assertIn(".", tokens)
        self.assertTrue(result["method"])

    def test_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            tag_text(" ")

    def test_unavailable_spacy_model_remains_useable(self):
        with patch(
            "modules.pos_tagging.load_spacy_model",
            return_value=(None, "The spaCy model 'en_core_web_sm' is not installed."),
        ), patch("nltk.pos_tag", side_effect=LookupError("missing averaged perceptron data")):
            result = tag_text("The curious student reads carefully.")

        self.assertTrue(result["tags"])
        self.assertTrue(result["method"])
        self.assertIn("rule-based fallback", result["method"])


if __name__ == "__main__":
    unittest.main()
