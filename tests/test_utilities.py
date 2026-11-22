import unittest
from unittest.mock import Mock, patch

from modules.statistics import analyze_text
from utils.file_handler import extract_document
from utils.helpers import history_csv, history_report, record_analysis
from utils.nlp_resources import is_wordnet_available


class UtilityTests(unittest.TestCase):
    def test_session_metrics_and_reports_use_recorded_analyses(self):
        state = {}
        record_analysis(state, "Statistics", "One two three.", "3 words")
        self.assertEqual(state["analysis_count"], 1)
        self.assertEqual(state["total_words_processed"], 3)
        self.assertIn("Statistics", history_csv(state["analysis_history"]))
        self.assertIn("3 words", history_report(state["analysis_history"]))

    def test_txt_reader_and_invalid_upload(self):
        self.assertEqual(extract_document("note.txt", b"Local document."), "Local document.")
        with self.assertRaises(ValueError):
            extract_document("archive.docx", b"not supported")
        with self.assertRaises(ValueError):
            extract_document("broken.pdf", b"not a PDF")

    def test_scanned_pdf_limitation_is_explicit(self):
        with patch("pypdf.PdfReader") as reader_type:
            reader = reader_type.return_value
            reader.is_encrypted = False
            reader.pages = [Mock(extract_text=Mock(return_value=""))]
            with self.assertRaisesRegex(ValueError, "OCR is not currently supported"):
                extract_document("scanned.pdf", b"%PDF-1.4\n%empty")

    def test_wordnet_availability_check_reports_missing_corpus(self):
        is_wordnet_available.cache_clear()
        with patch("nltk.data.find", side_effect=LookupError("missing")):
            available, reason = is_wordnet_available()

        self.assertFalse(available)
        self.assertIn("WordNet", reason)

    def test_statistics_fixture_is_usable_as_a_sample(self):
        sample = "NLP analyzes text. Text reveals patterns."
        self.assertEqual(analyze_text(sample)["sentence_count"], 2)


if __name__ == "__main__":
    unittest.main()
