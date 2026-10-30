"""
FAISS vector store persistence and management.
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from storage.paths import get_rag_db_path, is_faiss_index

logger = logging.getLogger(__name__)


def load_faiss_index(
    thread_id: str,
    embeddings: Embeddings,
) -> Optional[FAISS]:
    """Load a FAISS vector store for a specific conversation thread.
    
    Returns None if no valid index exists.
    """
    index_path = get_rag_db_path(thread_id)
    if not is_faiss_index(index_path):
        return None
    try:
        return FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        logger.error(f"Failed to load FAISS index at {index_path}: {e}")
        return None


def save_faiss_index(
    vector_store: FAISS,
    thread_id: str,
) -> Path:
    """Save a FAISS vector store for a specific conversation thread."""
    index_path = get_rag_db_path(thread_id)
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))
    return index_path

