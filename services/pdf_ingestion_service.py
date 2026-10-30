"""
PDF ingestion pipeline service: extraction, chunking, embedding, indexing, and manifest tracking.
Provides isolated storage per document_id, direct safe indexing, and PostgreSQL synchronization.
"""

from datetime import datetime, timezone
import gc
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import uuid

from langchain_community.vectorstores import FAISS as DefaultFAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from storage.paths import RAG_DATA_DIR, get_rag_db_path, is_faiss_index
from storage.document_metadata import (
    read_document_manifest as default_read_manifest,
    write_document_manifest as default_write_manifest,
    safe_remove_directory,
)
from tools.pdf import PdfIngestionError, extract_pdf_documents, sanitize_unicode
from services.embedding_service import check_ollama_health
from app.dependencies import get_embeddings

logger = logging.getLogger(__name__)


def ingest_rag_document(
    file_path: str,
    thread_id: str,
    filename: Optional[str] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    embeddings: Optional[Embeddings] = None,
    base_dir: Optional[Path] = None,
    faiss_cls: Optional[Any] = None,
    read_manifest_fn: Optional[Callable[[str], Any]] = None,
    write_manifest_fn: Optional[Callable[[str, Any], None]] = None,
    check_health: bool = True,
) -> Dict[str, Any]:
    """Extract, index, then persist an uploaded PDF under its conversation thread.
    
    Assigns each upload a unique document_id and isolated storage directory.
    Indexes directly into its document folder without staging renames, eliminating
    Windows directory rename lock errors completely.
    
    Returns:
        dict: The document manifest entry for the indexed file.
    Raises:
        PdfIngestionError: If extraction fails, text is empty, embedding service is unavailable, etc.
    """
    filename = sanitize_unicode(Path(filename or file_path).name)
    doc_id = str(uuid.uuid4())
    page_count = None
    character_count = 0
    emb = embeddings or get_embeddings()
    root_dir = base_dir if base_dir is not None else RAG_DATA_DIR
    faiss = faiss_cls if faiss_cls is not None else DefaultFAISS
    
    def _read_manifest(tid: str):
        if read_manifest_fn is not None:
            return read_manifest_fn(tid)
        return default_read_manifest(tid, base_dir=root_dir)

    def _write_manifest(tid: str, data: Any):
        if write_manifest_fn is not None:
            return write_manifest_fn(tid, data)
        return default_write_manifest(tid, data, base_dir=root_dir)

    final_path: Optional[Path] = None

    try:
        if progress_callback:
            progress_callback("Extracting", f"Extracting text from {filename}...")

        documents, page_count, character_count, page_errors = extract_pdf_documents(
            file_path,
            thread_id,
            filename,
        )

        # Sanitize all extracted documents
        for doc in documents:
            doc.page_content = sanitize_unicode(doc.page_content)
            doc.metadata["source"] = filename
            doc.metadata["thread_id"] = thread_id
            doc.metadata["doc_id"] = doc_id

        if progress_callback:
            progress_callback(
                "Extracting",
                f"Extracted {character_count:,} characters from {page_count} pages.",
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        # Re-sanitize every chunk to ensure 100% valid UTF-8 without surrogates or null bytes
        for chunk in chunks:
            chunk.page_content = sanitize_unicode(chunk.page_content)
            chunk.metadata["source"] = filename
            chunk.metadata["thread_id"] = thread_id
            chunk.metadata["doc_id"] = doc_id

        if not chunks:
            raise PdfIngestionError(
                "Text was extracted, but no usable chunks were produced for indexing."
            )

        # Validate embedding service health before beginning expensive indexing
        if check_health and hasattr(emb, "base_url"):
            if progress_callback:
                progress_callback("Validating", "Checking embedding service health...")
            check_ollama_health()

        thread_path = get_rag_db_path(thread_id, base_dir=root_dir)
        documents_path = thread_path / "documents"
        documents_path.mkdir(parents=True, exist_ok=True)
        final_path = documents_path / doc_id

        if progress_callback:
            progress_callback(
                "Indexing",
                f"Indexing {character_count:,} extracted characters across {len(chunks)} chunks...",
            )

        try:
            # Write directly to the unique isolated final_path
            vector_store = faiss.from_documents(chunks, emb)
            vector_store.save_local(str(final_path))
            
            # Explicitly drop reference and collect garbage
            del vector_store
            gc.collect()

            if not is_faiss_index(final_path):
                raise RuntimeError("FAISS did not create a complete index.")

            relative_index_path = str(final_path.relative_to(root_dir))
            extraction_method = "ocr" if any(d.metadata.get("extraction") == "ocr" for d in documents) else "native"
            document_entry = {
                "doc_id": doc_id,
                "filename": filename,
                "status": "ready",
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "page_count": page_count,
                "character_count": character_count,
                "chunk_count": len(chunks),
                "index_path": relative_index_path,
                "extraction_method": extraction_method,
            }
            if page_errors:
                skipped_pages = ", ".join(str(page) for page, _ in page_errors)
                document_entry["diagnostic"] = (
                    f"Text extraction failed on page(s) {skipped_pages}; "
                    "other readable pages were indexed."
                )

            manifest = _read_manifest(thread_id)
            # Remove any prior failed record for this filename or record with duplicate doc_id
            manifest = [
                m for m in manifest
                if m.get("doc_id") != doc_id
                and not (m.get("filename") == filename and m.get("status") == "failed")
            ]
            manifest.append(document_entry)

            if progress_callback:
                progress_callback("Indexing", "Saving document association...")
            _write_manifest(thread_id, manifest)

        except Exception:
            if final_path is not None and final_path.exists():
                safe_remove_directory(final_path)
            raise

        return document_entry

    except Exception as error:
        try:
            manifest = _read_manifest(thread_id)
            manifest = [
                m for m in manifest
                if m.get("doc_id") != doc_id
                and not (m.get("filename") == filename and m.get("status") == "failed")
            ]
            failed_entry = {
                "doc_id": doc_id,
                "filename": filename,
                "status": "failed",
                "error": str(error),
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "page_count": page_count,
                "character_count": 0,
                "chunk_count": 0,
            }
            manifest.append(failed_entry)
            _write_manifest(thread_id, manifest)
        except Exception:
            pass
        raise
