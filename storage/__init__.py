"""
Storage package for paths, document manifests, and FAISS vector indices.
"""

from storage.paths import (
    RAG_DATA_DIR,
    sanitize_thread_id,
    get_rag_db_path,
    get_manifest_path,
    get_documents_dir,
    is_faiss_index,
)
from storage.document_metadata import (
    read_document_manifest,
    write_document_manifest,
    record_document_status,
)
from storage.faiss_store import (
    load_faiss_index,
    save_faiss_index,
)

__all__ = [
    "RAG_DATA_DIR",
    "sanitize_thread_id",
    "get_rag_db_path",
    "get_manifest_path",
    "get_documents_dir",
    "is_faiss_index",
    "read_document_manifest",
    "write_document_manifest",
    "record_document_status",
    "load_faiss_index",
    "save_faiss_index",
]

