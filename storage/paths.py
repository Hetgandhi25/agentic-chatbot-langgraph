"""
Path management for vector stores and document storage.
"""

from pathlib import Path
from typing import Optional
import os


# Base directory for all conversation FAISS databases
RAG_DATA_DIR = Path(__file__).resolve().parent.parent / "faiss_db"


def sanitize_thread_id(thread_id: str) -> str:
    """Sanitize conversation thread ID for safe filesystem use."""
    safe_thread_id = "".join(
        char for char in str(thread_id) if char.isalnum() or char in ("-", "_")
    )
    if not safe_thread_id:
        raise ValueError("A valid conversation thread ID is required for PDF RAG.")
    return safe_thread_id


def get_rag_db_path(thread_id: str, base_dir: Optional[Path] = None) -> Path:
    """Return an isolated FAISS directory for one conversation."""
    root = base_dir if base_dir is not None else RAG_DATA_DIR
    return Path(root) / sanitize_thread_id(thread_id)


def get_manifest_path(thread_id: str, base_dir: Optional[Path] = None) -> Path:
    """Return the path to the document manifest for a conversation."""
    return get_rag_db_path(thread_id, base_dir=base_dir) / "documents.json"


def get_documents_dir(thread_id: str, base_dir: Optional[Path] = None) -> Path:
    """Return the path to the documents subdirectory for a conversation."""
    return get_rag_db_path(thread_id, base_dir=base_dir) / "documents"


def is_faiss_index(index_path: Path) -> bool:
    """Check whether a folder contains a valid FAISS index."""
    if not index_path.is_dir():
        return False
    return all(
        (index_path / filename).is_file()
        for filename in ("index.faiss", "index.pkl")
    )
