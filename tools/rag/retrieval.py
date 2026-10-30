"""
Vector retrieval and similarity scoring logic for RAG tool.
"""

from pathlib import Path
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def retrieve_relevant_documents(
    vector_store: FAISS,
    query: str,
    k: int = 4,
    score_threshold: float = 0.08,
) -> List[Document]:
    """Retrieve and score documents from FAISS index using relevance score threshold."""
    relevance_score_fn = vector_store._select_relevance_score_fn()
    scored_documents: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(query, k=k)
    return [
        document
        for document, distance in scored_documents
        if relevance_score_fn(float(distance)) >= score_threshold
    ]


def format_retrieved_documents(documents: List[Document]) -> str:
    """Format a list of retrieved LangChain Documents into a clean string for the LLM."""
    formatted_documents = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        source = Path(str(source)).name
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)

