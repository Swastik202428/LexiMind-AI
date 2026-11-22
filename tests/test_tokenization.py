import unittest

from modules.tokenization import ngram_counts, tokenize


class TokenizationTests(unittest.TestCase):
    def test_sentence_words_and_ngrams(self):
        result = tokenize("Cats chase mice. Dogs chase cats!")
        self.assertEqual(len(result["sentences"]), 2)
        self.assertEqual(result["word_count"], 6)
        counts = ngram_counts("Cats chase mice")
        self.assertEqual(counts["2-grams"]["cats chase"], 1)
        self.assertEqual(counts["3-grams"]["cats chase mice"], 1)

    def test_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            tokenize("")

    def test_rejects_unsupported_ngram_size(self):
        from modules.tokenization import ngrams

        with self.assertRaises(ValueError):
            ngrams("some words", 4)

    def test_hindi_sentence_and_words_remain_intact(self):
        result = tokenize("कृत्रिम बुद्धिमत्ता शिक्षा को बदल रही है। कृत्रिम बुद्धिमत्ता!")
        self.assertEqual(len(result["sentences"]), 2)
        self.assertIn("कृत्रिम", result["words"])
        self.assertIn("बुद्धिमत्ता", result["words"])
        self.assertNotIn("क", result["words"])
        self.assertEqual(result["word_count"], 9)

    def test_mixed_text_ngram_and_unicode_punctuation(self):
        from modules.tokenization import ngrams

        text = "Artificial Intelligence शिक्षा को बेहतर बना रही है।"
        result = tokenize(text)
        self.assertIn("Artificial", result["words"])
        self.assertIn("शिक्षा", result["words"])
        self.assertIn("artificial intelligence", " ".join(ngrams(text, 2)))


if __name__ == "__main__":
    unittest.main()
