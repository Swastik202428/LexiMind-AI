"""Local TXT and PDF extraction helpers."""

from io import BytesIO


def extract_document(filename: str, content: bytes) -> str:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix == "txt":
        text = content.decode("utf-8-sig", errors="replace")
    elif suffix == "pdf":
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
        except ImportError as error:
            raise RuntimeError("Install pypdf from requirements.txt to read PDF documents.") from error
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise ValueError("This PDF is password-protected and cannot be read.")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ValueError:
            raise
        except (PdfReadError, OSError, TypeError) as error:
            raise ValueError(f"Could not read this PDF: {error}") from error
    else:
        raise ValueError("Upload a .txt or .pdf file.")
    if not text.strip():
        raise ValueError(
            "No readable text was found. This may be a scanned PDF. "
            "OCR is not currently supported, so text could not be extracted."
        )
    return text
