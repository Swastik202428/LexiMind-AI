import unittest

from modules.summarization import summarize_text


class SummarizationTests(unittest.TestCase):
    def test_summary_is_extractive_and_respects_sentence_limit(self):
        text = (
            "The language model processes written language. "
            "It finds repeated terms in a document. "
            "Language models help researchers analyze written language."
        )
        result = summarize_text(text, sentence_count=2)
        self.assertEqual(len(result["sentences"]), 2)
        self.assertTrue(all(sentence in text for sentence in result["sentences"]))
        self.assertEqual(result["summary"], " ".join(result["sentences"]))

    def test_sentence_limit_is_bounded_by_available_sentences(self):
        result = summarize_text("Only one sentence.", sentence_count=8)
        self.assertEqual(len(result["sentences"]), 1)

    def test_rejects_non_positive_sentence_count(self):
        with self.assertRaises(ValueError):
            summarize_text("Only one sentence.", sentence_count=0)


if __name__ == "__main__":
    unittest.main()
