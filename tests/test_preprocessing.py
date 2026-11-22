import unittest
from unittest.mock import patch

from modules.preprocessing import process_text


class PreprocessingTests(unittest.TestCase):
    def test_normalizes_and_stems_words(self):
        result = process_text(
            "The cats are running! Version 2.",
            remove_stopwords=True,
            remove_numbers=True,
            stemming=True,
        )
        self.assertEqual(result["cleaned"], "cat run version")
        self.assertEqual(result["word_count_before"], 6)
        self.assertIn("Porter stemming", result["operations"])

    def test_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            process_text("  ")

    def test_punctuation_removal_preserves_token_boundaries(self):
        result = process_text(
            "alpha,beta",
            lowercase=False,
            remove_punctuation=True,
            remove_special_characters=False,
        )
        self.assertEqual(result["cleaned"], "alpha beta")

    def test_missing_wordnet_uses_rule_based_fallback(self):
        with patch(
            "modules.preprocessing.is_wordnet_available",
            return_value=(False, "The WordNet corpus is not installed."),
        ):
            result = process_text("The cats are running", lemmatization=True)

        self.assertIn("rule-based English lemmatization", result["operations"])
        self.assertEqual(result["cleaned"], "the cat are run")
        self.assertIn("WordNet corpus", result["resource_warning"])
        self.assertIn("suffix-based fallback", result["resource_warning"])

    def test_hindi_words_remain_intact_and_hindi_stem_is_used(self):
        result = process_text(
            "कृत्रिम बुद्धिमत्ता शिक्षा को बदल रही है।",
            stemming=True,
        )
        self.assertEqual(
            result["cleaned"],
            "कृत्रिम बुद्धिमत्ता शिक्षा को बदल रह है",
        )
        self.assertIn("rule-based Hindi suffix stemming", result["operations"])
        self.assertNotIn("Porter stemming", result["operations"])

    def test_mixed_text_preserves_hindi_and_stems_only_english(self):
        result = process_text(
            "Artificial Intelligence शिक्षा को बेहतर बना रही है।",
            stemming=True,
        )
        words = result["cleaned"].split()
        self.assertIn("शिक्षा", words)
        self.assertIn("artifici", words)
        self.assertIn("intellig", words)
        self.assertIn("Porter stemming", result["operations"])
        self.assertIn("rule-based Hindi suffix stemming", result["operations"])

    def test_hindi_stopword_resource_is_reported(self):
        result = process_text(
            "कृत्रिम बुद्धिमत्ता शिक्षा को बदल रही है।",
            remove_stopwords=True,
        )
        self.assertIn("Hindi built-in stopwords", result["operations"][0] if False else " ".join(result["operations"]))
        self.assertIn("Hindi built-in stopwords", " ".join(result["operations"]))
        self.assertNotIn("को", result["cleaned"].split())
        self.assertIn("कृत्रिम", result["cleaned"])

    def test_unicode_punctuation_numbers_and_repeated_words(self):
        result = process_text(
            "कृत्रिम—बुद्धिमत्ता! शिक्षा। शिक्षा 2026.",
            remove_numbers=True,
            remove_punctuation=True,
        )
        self.assertNotIn("2026", result["cleaned"])
        self.assertIn("कृत्रिम बुद्धिमत्ता", result["cleaned"])
        self.assertEqual(result["cleaned"].split().count("शिक्षा"), 2)


if __name__ == "__main__":
    unittest.main()
