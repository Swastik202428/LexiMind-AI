import unittest

from modules.statistics import analyze_text


class StatisticsTests(unittest.TestCase):
    def test_reports_counts_and_case_insensitive_frequency(self):
        result = analyze_text("One ONE two.")
        self.assertEqual(result["word_count"], 3)
        self.assertEqual(result["sentence_count"], 1)
        self.assertEqual(result["vocabulary_size"], 2)
        self.assertEqual(result["frequencies"][0], ("one", 2))
        self.assertEqual(result["average_word_length"], 3.0)

    def test_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            analyze_text(" ")

    def test_requires_a_positive_frequency_limit(self):
        with self.assertRaises(ValueError):
            analyze_text("Text has words.", limit=0)

    def test_hindi_words_sentences_and_frequencies(self):
        result = analyze_text("कृत्रिम बुद्धिमत्ता शिक्षा को बदल रही है।\nशिक्षा महत्वपूर्ण है।")
        self.assertEqual(result["word_count"], 10)
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(result["vocabulary_size"], 8)
        self.assertEqual(result["frequencies"][0], ("शिक्षा", 2))

    def test_mixed_text_counts_real_words(self):
        result = analyze_text("Artificial Intelligence शिक्षा को बेहतर बना रही है।")
        self.assertEqual(result["word_count"], 8)
        self.assertEqual(result["vocabulary_size"], 8)
        self.assertEqual(result["sentence_count"], 1)

    def test_punctuation_only_and_numbers(self):
        punctuation = analyze_text("!!!॥...")
        self.assertEqual(punctuation["word_count"], 0)
        numbers = analyze_text("2026 और 2026")
        self.assertEqual(numbers["word_count"], 3)
        self.assertEqual(numbers["frequencies"][0], ("2026", 2))


if __name__ == "__main__":
    unittest.main()
