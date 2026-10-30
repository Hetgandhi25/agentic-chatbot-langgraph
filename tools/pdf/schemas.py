"""
Schemas and exceptions for PDF ingestion and processing.
"""

from dataclasses import dataclass
from typing import Optional


class PdfIngestionError(ValueError):
    """Raised when an uploaded PDF cannot safely be extracted or indexed."""


def sanitize_unicode(text: str) -> str:
    """Sanitize text to guarantee valid UTF-8 and clean database/Ollama storage.

    1. Strips null bytes (\\x00) which crash PostgreSQL and C-based tokenizers.
    2. Re-encodes using UTF-16 surrogatepass to heal properly paired surrogate pairs (e.g. \\ud83d\\udca1 -> 💡).
    3. Encodes to UTF-8 with ignore to strip lone/unpaired surrogates (\\ud800-\\udfff) that trigger UnicodeEncodeError.
    """
    if not text:
        return ""
    text = text.replace("\x00", "")
    try:
        text = text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
    except Exception:
        pass
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore").replace("\x00", "")


@dataclass
class PdfDocumentInfo:
    doc_id: str
    filename: str
    status: str
    uploaded_at: str
    page_count: Optional[int] = None
    character_count: Optional[int] = None
    chunk_count: Optional[int] = None
    index_path: Optional[str] = None
    error: Optional[str] = None
    diagnostic: Optional[str] = None

