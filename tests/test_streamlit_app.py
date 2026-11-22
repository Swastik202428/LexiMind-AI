import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitNavigationTests(unittest.TestCase):
    def test_dashboard_and_all_navigation_pages_render(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=40).run()
        self.assertFalse(app.exception, str(app.exception))
        page_selector = app.radio(key="page")
        page_labels = list(page_selector.options)
        self.assertEqual(len(page_labels), 15)

        for label in page_labels[1:]:
            app.radio(key="page").set_value(label)
            app.run()
            self.assertFalse(app.exception, f"{label!r}: {app.exception}")

    def test_dashboard_shortcut_and_text_controls_work(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=40).run()
        app.button(key="open_Text Preprocessing").click()
        app.run()
        self.assertIn("Text Preprocessing", app.radio(key="page").value)

        app.button(key="use_example").click()
        app.run()
        self.assertIn("Natural language processing", app.text_area(key="source_text").value)

        app.button(key="clear_text").click()
        app.run()
        self.assertEqual(app.text_area(key="source_text").value, "")
        self.assertFalse(app.exception, str(app.exception))

    def test_primary_nlp_actions_render_results(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=40).run()
        labels = list(app.radio(key="page").options)
        sample = "OpenAI analyzes language text. Researchers use machine learning to find useful patterns."
        actions = [
            ("Text Preprocessing", "run_preprocessing", sample),
            ("Tokenization & N-Grams", "run_tokenization", sample),
            ("Text Statistics", "run_statistics", sample),
            ("Sentiment Analysis", "run_sentiment", "I love this excellent and helpful tool."),
            ("Named Entity Recognition", "run_ner", "OpenAI launched a product in London on March 5, 2025."),
            ("Text Classification", "run_classification", "Claim your free cash prize now."),
            ("Keyword Extraction", "run_keywords", sample),
            ("Text Summarization", "run_summary", sample),
            ("Language Detection", "run_language", "This is a longer English language detection sample."),
            ("POS Tagging", "run_pos", sample),
            ("Lexical Vectors", "run_embeddings", "Language tools analyze text.\nMachine learning finds patterns."),
            (
                "Topic Modeling",
                "run_topics",
                "Researchers study language data and text patterns.\n"
                "Businesses report quarterly revenue and market growth.\n"
                "Scientists publish laboratory research findings.",
            ),
            ("Question Answering", "run_qa", "LexiMind analyzes text with machine learning. It extracts useful keywords."),
        ]

        for title, action_key, text in actions:
            label = next(option for option in labels if title in option)
            app.radio(key="page").set_value(label)
            app.run()
            app.text_area(key="source_text").set_value(text)
            if title == "Question Answering":
                app.text_input(key="qa_question").set_value("What does LexiMind analyze?")
            app.button(key=action_key).click()
            app.run()
            self.assertFalse(app.exception, f"{title!r}: {app.exception}")
            self.assertIn(title, app.session_state["module_results"])


if __name__ == "__main__":
    unittest.main()
