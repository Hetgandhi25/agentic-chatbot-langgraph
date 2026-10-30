"""
Document discovery, validation, and retriever assembly services.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from storage.paths import RAG_DATA_DIR, get_rag_db_path, is_faiss_index
from storage.document_metadata import read_document_manifest, write_document_manifest
from app.dependencies import get_embeddings

logger = logging.getLogger(__name__)


def get_rag_documents(
    thread_id: str,
    status: Optional[str] = "ready",
    base_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return persistently registered PDFs for a thread, optionally filtered by status.
    
    status can be:
      - "ready": only successfully indexed and verified documents
      - "failed": only documents where extraction or indexing failed
      - "all" or None: all document records for the thread
    """
    root_dir = base_dir if base_dir is not None else RAG_DATA_DIR
    thread_path = get_rag_db_path(thread_id, base_dir=root_dir)
    documents: List[Dict[str, Any]] = []

    for entry in read_document_manifest(thread_id, base_dir=root_dir):
        entry_status = entry.get("status", "ready")
        if status is not None and status != "all" and entry_status != status:
            continue

        if entry_status == "ready":
            relative_index_path = entry.get("index_path")
            if not isinstance(relative_index_path, str):
                raise RuntimeError("A registered PDF is missing its index location.")

            index_path = (root_dir / relative_index_path).resolve()
            if not index_path.is_relative_to(root_dir.resolve()):
                raise RuntimeError("A registered PDF has an invalid index location.")
            if not is_faiss_index(index_path):
                raise RuntimeError(
                    f"The saved FAISS index for {entry.get('filename', 'a PDF')} "
                    "is missing or incomplete. Please upload the PDF again."
                )
        documents.append(entry)

    # Older versions stored one index directly in the thread directory.
    if (status is None or status in ("ready", "all")) and is_faiss_index(thread_path):
        if not any(d.get("doc_id") == "legacy" for d in documents):
            documents.append({
                "doc_id": "legacy",
                "filename": "Previously indexed PDF (filename unavailable)",
                "status": "ready",
                "index_path": str(thread_path.relative_to(root_dir)),
                "page_count": None,
                "character_count": None,
                "chunk_count": None,
            })

    return documents


def has_rag_index(thread_id: str, base_dir: Optional[Path] = None) -> bool:
    """Check whether a valid ready FAISS index exists for this conversation."""
    return bool(get_rag_documents(thread_id, status="ready", base_dir=base_dir))


def has_thread_documents(thread_id: str, base_dir: Optional[Path] = None) -> bool:
    """Check whether any documents (ready or failed) are associated with this thread."""
    return bool(get_rag_documents(thread_id, status=None, base_dir=base_dir))


def clear_failed_documents(thread_id: str, base_dir: Optional[Path] = None) -> None:
    """Clear failed document entries from this thread's manifest so the user can retry cleanly."""
    root_dir = base_dir if base_dir is not None else RAG_DATA_DIR
    manifest = read_document_manifest(thread_id, base_dir=root_dir)
    updated = [d for d in manifest if d.get("status") != "failed"]
    write_document_manifest(thread_id, updated, base_dir=root_dir)


def get_retriever(
    thread_id: str,
    embeddings: Optional[Embeddings] = None,
    base_dir: Optional[Path] = None,
    target_doc_id: Optional[str] = None,
):
    """Load and merge persisted PDF FAISS indexes for the given conversation thread.

    If target_doc_id is provided, only loads and returns the index for that specific document.
    Otherwise loads all ready documents and merges them for cross-document retrieval.
    """
    root_dir = base_dir if base_dir is not None else RAG_DATA_DIR
    documents = get_rag_documents(thread_id, status="ready", base_dir=root_dir)
    if not documents:
        return None

    if target_doc_id:
        documents = [d for d in documents if d.get("doc_id") == target_doc_id]
        if not documents:
            return None

    emb = embeddings or get_embeddings()
    vector_stores = []
    for document in documents:
        idx_rel = document.get("index_path")
        if not idx_rel:
            continue
        p = root_dir / idx_rel
        if is_faiss_index(p):
            try:
                vs = FAISS.load_local(
                    folder_path=str(p),
                    embeddings=emb,
                    allow_dangerous_deserialization=True,
                )
                vector_stores.append(vs)
            except Exception as e:
                logger.warning(f"Could not load vector store at {p}: {e}")

    if not vector_stores:
        return None

    vector_store = vector_stores[0]
    for additional_store in vector_stores[1:]:
        vector_store.merge_from(additional_store)

    # When multiple documents are present, retrieve up to 12 chunks across them
    k = min(12, max(4, 3 * len(vector_stores)))
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
