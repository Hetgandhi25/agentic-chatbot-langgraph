"""
Unit tests for PDF extraction module.
"""

import os
import tempfile
import unittest
from tools.pdf import PdfIngestionError, check_ocr_availability, extract_pdf_documents


class TestPdfExtraction(unittest.TestCase):
    def test_ocr_availability_contract(self):
        is_avail, reason = check_ocr_availability()
        self.assertIsInstance(is_avail, bool)
        self.assertIsInstance(reason, str)

    def test_missing_file_raises_error(self):
        with self.assertRaises(PdfIngestionError) as ctx:
            extract_pdf_documents("nonexistent_file_xyz.pdf", "thread-1", "nonexistent_file_xyz.pdf")
        self.assertIn("was not found", str(ctx.exception))

    def test_empty_file_raises_error(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"")
            path = f.name
        try:
            with self.assertRaises(PdfIngestionError) as ctx:
                extract_pdf_documents(path, "thread-1", "empty.pdf")
            self.assertIn("empty", str(ctx.exception).lower())
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_invalid_signature_raises_error(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"NOT_A_PDF_CONTENT")
            path = f.name
        try:
            with self.assertRaises(PdfIngestionError) as ctx:
                extract_pdf_documents(path, "thread-1", "invalid.pdf")
            self.assertIn("missing its pdf signature", str(ctx.exception).lower())
        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()

