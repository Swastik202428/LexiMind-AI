import unittest

from modules.classification import (
    classify_text,
    evaluate_models,
    load_training_examples,
)


class ClassificationTests(unittest.TestCase):
    def test_spam_classifier_returns_a_normalized_probability_distribution(self):
        result = classify_text("Claim your free cash prize now", "spam")
        self.assertEqual(result["label"], "Spam")
        self.assertAlmostEqual(sum(result["scores"].values()), 1.0, places=6)
        self.assertGreater(result["training_examples"], 0)

    def test_news_classifier_uses_the_expected_labels(self):
        result = classify_text("The team won the football match", "news")
        self.assertEqual(result["label"], "Sports")
        self.assertIn("Sports", result["scores"])

    def test_rejects_unknown_task(self):
        with self.assertRaises(ValueError):
            classify_text("A document", "unknown")

    def test_included_datasets_have_multiple_labels(self):
        for task in ("spam", "news"):
            examples = load_training_examples(task)
            self.assertGreater(len(examples), 10)
            self.assertGreaterEqual(len({label for _, label in examples}), 2)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            classify_text("  ", "spam")

    def test_held_out_evaluation_is_reproducible_and_complete(self):
        first = evaluate_models("news")
        second = evaluate_models("news")
        self.assertEqual(first, second)
        self.assertEqual(first["split"]["train"] + first["split"]["validation"] + first["split"]["test"], len(load_training_examples("news")))
        self.assertIn("accuracy", first["test"])
        self.assertIn("precision_macro", first["test"])
        self.assertIn("recall_macro", first["test"])
        self.assertIn("macro_f1", first["test"])
        self.assertIn("confusion_matrix", first["test"])
        self.assertEqual(set(first["test"]["per_class"]), set(first["test"]["class_labels"]))

    def test_imbalanced_examples_are_rejected_when_split_is_not_stratifiable(self):
        examples = [
            ("A first politics story", "Politics"),
            ("A second politics story", "Politics"),
            ("A sports championship", "Sports"),
        ]
        with self.assertRaisesRegex(ValueError, "(?i)(at least four|at least five|five)"):
            evaluate_models("news", examples=examples)

    def test_duplicate_examples_are_rejected(self):
        examples = [
            ("The same article", "Politics"),
            ("The same article", "Sports"),
            ("Another politics article", "Politics"),
            ("Another sports article", "Sports"),
            ("A third politics article", "Politics"),
            ("A third sports article", "Sports"),
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_models("news", examples=examples)


if __name__ == "__main__":
    unittest.main()
