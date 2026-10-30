"""
PDF text extraction engine with vector-path detection and OCR fallback.
"""

import logging
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader
from langchain_core.documents import Document

from tools.pdf.schemas import PdfIngestionError, sanitize_unicode
from tools.pdf.ocr import check_ocr_availability, try_ocr_extraction

logger = logging.getLogger(__name__)



def extract_pdf_documents(
    file_path: str,
    thread_id: str,
    filename: str,
) -> Tuple[List[Document], int, int, List[Tuple[int, str]]]:
    """Extract text documents from a PDF file with complete validation and OCR fallback.
    
    Fallback strategy:
      1. Extract native selectable text first.
      2. If extracted text is empty or insufficient (< 40 characters for multi-page documents),
         render pages and run Tesseract OCR.
      3. Return extracted documents with character count and metadata.
    
    Raises:
        PdfIngestionError: If extraction fails, file is corrupted/encrypted, or 0 text characters found.
    """
    path = Path(file_path)
    try:
        if not path.is_file():
            raise PdfIngestionError(f"The PDF file '{filename}' was not found.")
        if path.stat().st_size == 0:
            raise PdfIngestionError("The uploaded PDF is empty.")
        with path.open("rb") as pdf_file:
            signature = pdf_file.read(5)
        if signature != b"%PDF-":
            raise PdfIngestionError(
                "The uploaded file is not a valid PDF (missing its PDF signature)."
            )

        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            raise PdfIngestionError(
                "This PDF is password-protected. Upload an unlocked copy; "
                "password-protected PDFs are not supported."
            )
        pages = reader.pages
        page_count = len(pages)
    except PdfIngestionError:
        raise
    except Exception as error:
        raise PdfIngestionError(
            f"Could not open the PDF ({type(error).__name__}): {error}"
        ) from error

    if page_count == 0:
        raise PdfIngestionError("The PDF contains no pages.")

    page_label = "page" if page_count == 1 else "pages"
    documents: List[Document] = []
    image_only_pages: List[int] = []
    page_errors: List[Tuple[int, str]] = []
    image_check_errors: List[Tuple[int, str]] = []
    character_count = 0

    clean_filename = sanitize_unicode(filename)

    # 1. Native text extraction
    for page_number, page in enumerate(pages, start=1):
        try:
            raw_text = page.extract_text() or ""
            text = sanitize_unicode(raw_text).strip()
        except Exception as error:
            page_errors.append((page_number, type(error).__name__))
            text = ""

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": clean_filename,
                        "page": page_number,
                        "thread_id": thread_id,
                        "extraction": "native",
                    },
                )
            )
            character_count += len(text)
            continue

        try:
            if hasattr(page, "images") and len(page.images) > 0:
                image_only_pages.append(page_number)
        except Exception as error:
            image_check_errors.append((page_number, type(error).__name__))

    # 2. If native extraction produced insufficient or 0 text, run OCR fallback
    insufficient_text = (len(documents) == 0) or (page_count > 1 and character_count < 50)

    if insufficient_text:
        ocr_available, ocr_reason = check_ocr_availability()

        if ocr_available:
            try:
                ocr_docs, ocr_chars = try_ocr_extraction(
                    file_path=str(path),
                    filename=filename,
                    thread_id=thread_id,
                    pages=pages,
                )
                if ocr_docs and ocr_chars > character_count:
                    return ocr_docs, page_count, ocr_chars, page_errors
            except Exception as e:
                logger.warning(f"OCR fallback failed on {filename}: {e}")

        # If OCR was not available or produced 0 text, diagnose root cause and report
        if not documents:
            # Check for vector paths / outlined text (e.g. from Print to PDF)
            has_vector_paths = False
            try:
                for p in pages:
                    contents = p.get("/Contents")
                    data = b""
                    if isinstance(contents, list):
                        data = b"".join(c.get_data() for c in contents if hasattr(c, "get_data"))
                    elif contents and hasattr(contents, "get_data"):
                        data = contents.get_data()
                    if b" c\n" in data or b" l\n" in data or b" m\n" in data:
                        has_vector_paths = True
                        break
            except Exception:
                pass

            if not ocr_available:
                if image_only_pages:
                    raise PdfIngestionError(
                        f"The PDF has {page_count} {page_label} and 0 extracted text characters. "
                        f"No selectable text was found, but image-only/scanned pages were detected ({len(image_only_pages)} pages). "
                        f"OCR is unavailable: {ocr_reason} Please install Tesseract OCR or upload a text-searchable PDF."
                    )

                if has_vector_paths:
                    raise PdfIngestionError(
                        f"The PDF has {page_count} {page_label} and 0 extracted text characters. "
                        "The file contains vector-drawn / outlined text without selectable font text (often produced by 'Print to PDF' or graphics exports). "
                        f"OCR is unavailable: {ocr_reason} Please upload a text-searchable PDF or install Tesseract OCR."
                    )

                raise PdfIngestionError(
                    f"The PDF has {page_count} {page_label} and 0 extracted text characters. "
                    f"OCR is unavailable: {ocr_reason} It may be blank, image-only, or contain outlined text."
                )

            # If OCR was available but still extracted 0 characters, the PDF is truly blank
            raise PdfIngestionError(
                f"The PDF has {page_count} {page_label} and 0 extracted text characters. "
                "Both native text extraction and OCR produced no readable text. The document appears to be blank."
            )

    return documents, page_count, character_count, page_errors
