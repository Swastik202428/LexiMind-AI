import unittest

from modules.question_answering import answer_question


class QuestionAnsweringTests(unittest.TestCase):
    def test_retrieves_the_most_relevant_context_sentence(self):
        context = (
            "LexiMind analyzes text with machine learning. "
            "The platform creates reports for users."
        )
        result = answer_question(context, "What does LexiMind analyze?")
        self.assertEqual(result["answer"], "LexiMind analyzes text with machine learning.")
        self.assertGreater(result["relevance"], 0)
        self.assertIn("not a generative LLM", result["method"])

    def test_validates_context_and_question(self):
        with self.assertRaises(ValueError):
            answer_question("", "What is this?")
        with self.assertRaises(ValueError):
            answer_question("A context sentence.", " ")

    def test_does_not_return_an_unrelated_sentence_as_an_answer(self):
        result = answer_question(
            "The library closes at six. The team practices on Tuesday.",
            "How does photosynthesis produce glucose?",
        )
        self.assertFalse(result["supported"])
        self.assertIsNone(result["answer"])


if __name__ == "__main__":
    unittest.main()
