import unittest

from modules.chatbot import respond


class ChatbotTests(unittest.TestCase):
    def test_greeting_and_known_faq(self):
        self.assertEqual(respond("hello")["intent"], "greeting")
        result = respond("What is tokenization?")
        self.assertEqual(result["intent"], "nlp_faq")
        self.assertIn("Tokenization splits", result["reply"])

    def test_unknown_intent_is_transparent(self):
        result = respond("What is the weather on Mars tomorrow?")
        self.assertEqual(result["intent"], "fallback")
        self.assertIn("not an LLM", result["reply"])

    def test_rejects_empty_message(self):
        with self.assertRaises(ValueError):
            respond(" ")


if __name__ == "__main__":
    unittest.main()
