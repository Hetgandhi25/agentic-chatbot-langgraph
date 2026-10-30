"""
OCR detection and text extraction support for image-only, scanned, or outlined-text PDFs.
"""

import io
import logging
import os
from pathlib import Path
import shutil
from typing import List, Optional, Tuple
from langchain_core.documents import Document

from tools.pdf.schemas import sanitize_unicode

logger = logging.getLogger(__name__)


def get_tesseract_command_path() -> Optional[str]:
    """Find Tesseract executable, checking custom D: drive path first, then PATH, then standard paths."""
    # 1. Environment variable override
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and os.path.isfile(env_cmd):
        return env_cmd

    # 2. Project dedicated D: drive path
    project_tesseract = (
        Path(__file__).resolve().parent.parent.parent
        / "resources"
        / "ocr"
        / "tesseract"
        / "tesseract.exe"
    )
    if project_tesseract.is_file():
        return str(project_tesseract)

    # 3. In PATH
    which_cmd = shutil.which("tesseract")
    if which_cmd:
        return which_cmd

    # 4. Standard Windows locations
    for std_path in (
        r"D:\resources\ocr\tesseract\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ):
        if os.path.isfile(std_path):
            return std_path

    return None


def get_tessdata_path(tesseract_binary: Optional[str] = None) -> Optional[str]:
    """Find tessdata language directory on D: drive or standard paths."""
    env_tessdata = os.getenv("TESSDATA_PREFIX")
    if env_tessdata and os.path.isdir(env_tessdata):
        return env_tessdata

    project_tessdata = (
        Path(__file__).resolve().parent.parent.parent
        / "resources"
        / "ocr"
        / "tesseract"
        / "tessdata"
    )
    if project_tessdata.is_dir():
        return str(project_tessdata)

    if tesseract_binary:
        bin_tessdata = Path(tesseract_binary).parent / "tessdata"
        if bin_tessdata.is_dir():
            return str(bin_tessdata)

    return None


def check_ocr_availability() -> Tuple[bool, str]:
    """Check whether a usable OCR engine and rasterizer are installed."""
    try:
        import pytesseract
    except ImportError:
        return False, "Tesseract OCR (pytesseract) is not installed in the Python environment."

    tesseract_cmd = get_tesseract_command_path()
    if not tesseract_cmd:
        return (
            False,
            "pytesseract is installed, but the Tesseract OCR binary was not found in "
            "D:\\Agentic-Chatbot-using-LangGraph\\resources\\ocr\\tesseract, PATH, or standard installation directories."
        )

    tessdata_dir = get_tessdata_path(tesseract_cmd)
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    if tessdata_dir:
        os.environ["TESSDATA_PREFIX"] = tessdata_dir

    return True, f"Tesseract OCR is available at {tesseract_cmd}."


def try_ocr_extraction(
    file_path: str,
    filename: str,
    thread_id: str,
    pages=None,
) -> Tuple[List[Document], int]:
    """
    Attempt OCR text extraction using PDF page rendering (pypdfium2 / pdf2image)
    or embedded images as fallback.
    """
    ocr_available, _ = check_ocr_availability()
    if not ocr_available:
        return [], 0

    import pytesseract

    clean_filename = sanitize_unicode(filename)

    # Method 1: pypdfium2 (Direct PDF page rendering to PIL images without external dependencies)
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(file_path)
        ocr_documents: List[Document] = []
        ocr_character_count = 0

        for page_number, page in enumerate(pdf, start=1):
            # Render at 2.0 scale (approx 144 DPI)
            pil_img = page.render(scale=2.0).to_pil()
            raw_text = pytesseract.image_to_string(pil_img) or ""
            extracted_text = sanitize_unicode(raw_text).strip()
            if extracted_text:
                ocr_documents.append(
                    Document(
                        page_content=extracted_text,
                        metadata={
                            "source": clean_filename,
                            "page": page_number,
                            "thread_id": thread_id,
                            "extraction": "ocr",
                        },
                    )
                )
                ocr_character_count += len(extracted_text)

        if ocr_documents:
            return ocr_documents, ocr_character_count
    except Exception as e:
        logger.warning(f"pypdfium2 OCR extraction failed: {e}")

    # Method 2: pdf2image (if poppler is installed)
    try:
        from pdf2image import convert_from_path

        temp_dir = str(Path(__file__).resolve().parent.parent.parent / ".tmp")
        images = convert_from_path(file_path, dpi=150, output_folder=temp_dir)
        ocr_documents = []
        ocr_character_count = 0

        for page_number, pil_img in enumerate(images, start=1):
            raw_text = pytesseract.image_to_string(pil_img) or ""
            extracted_text = sanitize_unicode(raw_text).strip()
            if extracted_text:
                ocr_documents.append(
                    Document(
                        page_content=extracted_text,
                        metadata={
                            "source": clean_filename,
                            "page": page_number,
                            "thread_id": thread_id,
                            "extraction": "ocr",
                        },
                    )
                )
                ocr_character_count += len(extracted_text)

        if ocr_documents:
            return ocr_documents, ocr_character_count
    except Exception as e:
        logger.warning(f"pdf2image OCR extraction failed: {e}")

    # Method 3: Embedded raster images fallback from pypdf pages
    if pages:
        try:
            from PIL import Image

            ocr_documents = []
            ocr_character_count = 0

            for page_number, page in enumerate(pages, start=1):
                page_text = ""
                if hasattr(page, "images") and len(page.images) > 0:
                    for img in page.images:
                        try:
                            pil_img = Image.open(io.BytesIO(img.data))
                            extracted = pytesseract.image_to_string(pil_img) or ""
                            page_text += " " + sanitize_unicode(extracted).strip()
                        except Exception:
                            pass
                page_text = sanitize_unicode(page_text).strip()
                if page_text:
                    ocr_documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": clean_filename,
                                "page": page_number,
                                "thread_id": thread_id,
                                "extraction": "ocr",
                            },
                        )
                    )
                    ocr_character_count += len(page_text)

            if ocr_documents:
                return ocr_documents, ocr_character_count
        except Exception as e:
            logger.warning(f"Embedded image OCR extraction failed: {e}")

    return [], 0
