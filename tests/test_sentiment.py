import unittest

from modules.sentiment import analyze_sentiment


class SentimentTests(unittest.TestCase):
    def test_polarity_signs_match_obvious_examples(self):
        self.assertEqual(analyze_sentiment("I love this excellent result.")["label"], "Positive")
        self.assertEqual(analyze_sentiment("I hate this terrible result.")["label"], "Negative")
        self.assertEqual(analyze_sentiment("The report contains three sections.")["label"], "Neutral")

    def test_scores_are_bounded(self):
        result = analyze_sentiment("A useful and thoughtful example.")
        self.assertGreaterEqual(result["polarity"], -1)
        self.assertLessEqual(result["polarity"], 1)
        self.assertGreaterEqual(result["subjectivity"], 0)
        self.assertLessEqual(result["subjectivity"], 1)


if __name__ == "__main__":
    unittest.main()
