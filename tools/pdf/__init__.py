"""
PDF tools package for extraction, OCR, and validation.
"""

from tools.pdf.schemas import PdfIngestionError, PdfDocumentInfo
from tools.pdf.ocr import check_ocr_availability, try_ocr_extraction
from tools.pdf.extractor import extract_pdf_documents, sanitize_unicode

__all__ = [
    "PdfIngestionError",
    "PdfDocumentInfo",
    "check_ocr_availability",
    "try_ocr_extraction",
    "extract_pdf_documents",
    "sanitize_unicode",
]

