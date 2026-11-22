import unittest
from unittest.mock import patch

from modules.ner import _regex_entities, extract_entities


class EntityRecognitionTests(unittest.TestCase):
    def test_fallback_identifies_common_entity_types_and_offsets(self):
        text = "OpenAI paid $50 in London on March 5, 2025."
        entities = _regex_entities(text)
        found = {(item["text"], item["label"]) for item in entities}
        self.assertIn(("OpenAI", "ORG"), found)
        self.assertIn(("London", "LOCATION"), found)
        self.assertIn(("$50", "MONEY"), found)
        self.assertIn(("March 5, 2025", "DATE"), found)
        for item in entities:
            self.assertEqual(text[item["start"] : item["end"]], item["text"])

    def test_unavailable_spacy_model_uses_named_fallback(self):
        with patch(
            "modules.ner.load_spacy_model",
            return_value=(None, "The spaCy model 'en_core_web_sm' is not installed."),
        ):
            result = extract_entities("OpenAI paid $50 in London.")

        self.assertTrue(result["fallback"])
        self.assertIn("local pattern-based fallback", result["method"])


if __name__ == "__main__":
    unittest.main()
