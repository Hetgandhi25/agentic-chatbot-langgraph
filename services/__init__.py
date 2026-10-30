"""
Services package orchestrating business logic for LLMs, embeddings, PDFs, and conversations.
"""

from services.embedding_service import (
    EmbeddingServiceError,
    get_effective_ollama_config,
    check_ollama_health,
)
from services.llm_service import (
    get_llm_service,
    get_llm_with_tools,
)
from services.document_service import (
    get_rag_documents,
    has_rag_index,
    has_thread_documents,
    clear_failed_documents,
    get_retriever,
)
from services.pdf_ingestion_service import (
    ingest_rag_document,
)
from services.conversation_service import (
    get_all_threads,
)

__all__ = [
    "EmbeddingServiceError",
    "get_effective_ollama_config",
    "check_ollama_health",
    "get_llm_service",
    "get_llm_with_tools",
    "get_rag_documents",
    "has_rag_index",
    "has_thread_documents",
    "clear_failed_documents",
    "get_retriever",
    "ingest_rag_document",
    "get_all_threads",
]

