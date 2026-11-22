import unittest

from modules.language import detect_language


class LanguageTests(unittest.TestCase):
    def test_detects_english(self):
        result = detect_language("This is a sufficiently long English sentence for detection.")
        self.assertEqual(result["code"], "en")
        self.assertEqual(result["name"], "English")
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertEqual(result["candidates"][0]["code"], "en")

    def test_handles_short_devanagari_input(self):
        result = detect_language("\u0928\u092e\u0938\u094d\u0924\u0947")
        self.assertEqual(result["code"], "hi")

    def test_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            detect_language("")


if __name__ == "__main__":
    unittest.main()
