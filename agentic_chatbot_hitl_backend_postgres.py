"""
Backward-compatibility facade for Agentic Chatbot HITL Backend with PostgreSQL.

This module re-exports all graph objects, tools, services, storage utilities,
and database connections from the modular packages (agent, tools, services, database, storage).
Existing scripts and tests can import from here without breaking changes.
"""

from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple
import uuid

import psycopg
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from pypdf import PdfReader
import requests

# Modular imports
from app.config import settings, normalize_ollama_url
from app.dependencies import (
    get_llm,
    get_embeddings,
    get_db_connection,
    get_checkpointer,
)
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
from services.embedding_service import (
    EmbeddingServiceError,
    get_effective_ollama_config,
    check_ollama_health as service_check_ollama_health,
)
from services.document_service import (
    get_rag_documents as service_get_rag_documents,
    has_rag_index as service_has_rag_index,
    has_thread_documents as service_has_thread_documents,
    clear_failed_documents as service_clear_failed_documents,
    get_retriever as service_get_retriever,
)
from services.pdf_ingestion_service import (
    ingest_rag_document as service_ingest_rag_document,
)
from services.conversation_service import (
    get_all_threads as service_get_all_threads,
)
from tools.pdf import (
    PdfIngestionError,
    check_ocr_availability,
    extract_pdf_documents,
)
from tools.search import search_tool
from tools.calculator import calculator
from tools.stock_price import get_stock_price
from tools.weather import get_current_weather
from tools.purchase_stock import purchase_stock
from tools import get_all_tools
from tools.rag.retrieval import retrieve_relevant_documents, format_retrieved_documents
from agent.state import ChatState
from agent.prompts import SYSTEM_PROMPT_TEXT, get_system_message
from agent.routing import _get_deterministic_tool_call

# Module-level singletons (can be patched by test fixtures)
llm = get_llm()
embeddings = get_embeddings()
postgres_conn = get_db_connection()

OLLAMA_RAW_URL, OLLAMA_MODEL_NAME = get_effective_ollama_config()
OLLAMA_EFFECTIVE_URL = normalize_ollama_url(OLLAMA_RAW_URL)
_normalize_ollama_url = normalize_ollama_url


def check_ollama_health(timeout: float = 5.0) -> Dict[str, Any]:
    """Verify that the Ollama embedding service is running and the configured model is installed."""
    return service_check_ollama_health(
        timeout=timeout,
        config_getter=get_effective_ollama_config,
    )


# Path and manifest helpers adhering to module-level RAG_DATA_DIR
def _rag_db_path(thread_id: str) -> str:
    """Return an isolated FAISS directory for one conversation."""
    safe_thread_id = sanitize_thread_id(thread_id)
    return str(RAG_DATA_DIR / safe_thread_id)


def _manifest_path(thread_id: str) -> Path:
    """Return the document manifest path."""
    return Path(_rag_db_path(thread_id)) / "documents.json"


def _is_faiss_index(index_path: Path) -> bool:
    """Check whether a folder contains a valid FAISS index."""
    return is_faiss_index(index_path)


def _read_document_manifest(thread_id: str) -> List[Dict[str, Any]]:
    """Read document manifest list for a conversation."""
    path = _manifest_path(thread_id)
    if not path.exists():
        return []

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read the PDF document manifest for this chat: {error}"
        ) from error

    if not isinstance(manifest, dict) or not isinstance(manifest.get("documents"), list):
        raise RuntimeError("The PDF document manifest has an invalid format.")
    if not all(isinstance(entry, dict) for entry in manifest["documents"]):
        raise RuntimeError("The PDF document manifest contains an invalid entry.")
    return manifest["documents"]


def _write_document_manifest(thread_id: str, documents: List[Dict[str, Any]]) -> None:
    """Write document manifest list for a conversation."""
    thread_path = Path(_rag_db_path(thread_id))
    thread_path.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(thread_id)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=thread_path,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump({"documents": documents}, temporary_file, ensure_ascii=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def get_rag_documents(thread_id: str, status: Optional[str] = "ready") -> List[Dict[str, Any]]:
    """Return persistently registered PDFs for a thread, optionally filtered by status."""
    thread_path = Path(_rag_db_path(thread_id))
    documents: List[Dict[str, Any]] = []

    for entry in _read_document_manifest(thread_id):
        entry_status = entry.get("status", "ready")
        if status is not None and status != "all" and entry_status != status:
            continue

        if entry_status == "ready":
            relative_index_path = entry.get("index_path")
            if not isinstance(relative_index_path, str):
                raise RuntimeError("A registered PDF is missing its index location.")

            index_path = (RAG_DATA_DIR / relative_index_path).resolve()
            if not index_path.is_relative_to(RAG_DATA_DIR.resolve()):
                raise RuntimeError("A registered PDF has an invalid index location.")
            if not _is_faiss_index(index_path):
                raise RuntimeError(
                    f"The saved FAISS index for {entry.get('filename', 'a PDF')} "
                    "is missing or incomplete. Please upload the PDF again."
                )
        documents.append(entry)

    # Older versions stored one index directly in the thread directory.
    if (status is None or status in ("ready", "all")) and _is_faiss_index(thread_path):
        if not any(d.get("doc_id") == "legacy" for d in documents):
            documents.append({
                "doc_id": "legacy",
                "filename": "Previously indexed PDF (filename unavailable)",
                "status": "ready",
                "index_path": str(thread_path.relative_to(RAG_DATA_DIR)),
                "page_count": None,
                "character_count": None,
                "chunk_count": None,
            })

    return documents


def has_rag_index(thread_id: str) -> bool:
    """Check whether a valid ready FAISS index exists for this conversation."""
    return bool(get_rag_documents(thread_id, status="ready"))


def has_thread_documents(thread_id: str) -> bool:
    """Check whether any documents (ready or failed) are associated with this thread."""
    return bool(get_rag_documents(thread_id, status=None))


def clear_failed_documents(thread_id: str) -> None:
    """Clear failed document entries from this thread's manifest so the user can retry cleanly."""
    manifest = _read_document_manifest(thread_id)
    updated = [d for d in manifest if d.get("status") != "failed"]
    _write_document_manifest(thread_id, updated)


def _extract_pdf_documents(
    file_path: str,
    thread_id: str,
    filename: str,
) -> Tuple[List[Document], int, int, List[Tuple[int, str]]]:
    """Extract documents using tools.pdf.extract_pdf_documents."""
    return extract_pdf_documents(file_path, thread_id, filename)


def ingest_rag_document(
    file_path: str,
    thread_id: str,
    filename: Optional[str] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Extract, index, then persist an uploaded PDF under its conversation thread.
    
    Delegates to services.pdf_ingestion_service for isolated document storage,
    Windows-safe atomic indexing, and PostgreSQL metadata synchronization.
    """
    return service_ingest_rag_document(
        file_path=file_path,
        thread_id=thread_id,
        filename=filename,
        progress_callback=progress_callback,
        embeddings=embeddings,
        base_dir=RAG_DATA_DIR,
        faiss_cls=FAISS,
        read_manifest_fn=_read_document_manifest,
        write_manifest_fn=_write_document_manifest,
    )




def get_retriever(thread_id: str, target_doc_id: Optional[str] = None):
    """Load all persisted PDF indexes associated with the current conversation."""
    return service_get_retriever(thread_id, embeddings=embeddings, base_dir=RAG_DATA_DIR, target_doc_id=target_doc_id)



@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """
    Retrieve information only from PDFs uploaded in the current conversation.

    Use this tool for questions about an uploaded PDF/document. Each chat has
    its own FAISS index, so documents from other conversations are not used.

    Args:
        query: The question or search query used to retrieve PDF content.
        config: LangGraph runtime configuration used to identify the chat.
    """
    thread_id = (config or {}).get("configurable", {}).get("thread_id")

    if not thread_id:
        return (
            "I can't search a PDF because this conversation has no valid "
            "thread ID. Please start a new chat and upload the PDF again."
        )

    ready_docs = get_rag_documents(thread_id, status="ready")
    failed_docs = get_rag_documents(thread_id, status="failed")

    lower_query = query.lower()

    # Detect if query refers to the newest / latest / recent PDF
    new_doc_keywords = (
        "new pdf", "new document", "new doc", "new file",
        "latest pdf", "latest document", "latest doc", "latest file",
        "recent pdf", "recent document", "last pdf", "last document",
        "3rd pdf", "third pdf", "2nd pdf", "second pdf",
    )
    is_asking_for_new_doc = any(phrase in lower_query for phrase in new_doc_keywords)

    # 1. Check if user is asking for the new/latest document while the latest upload failed
    if failed_docs and is_asking_for_new_doc:
        latest_failed = failed_docs[-1]
        filename = latest_failed.get("filename", "the uploaded document")
        failure_reason = latest_failed.get("error", "Text extraction failed.")
        ready_names = [d.get("filename", "Document") for d in ready_docs]
        ready_context = f" Earlier indexed document(s) available in this chat: {', '.join(ready_names)}." if ready_names else ""
        return (
            f"The newly uploaded PDF ('{filename}') could not be processed: {failure_reason} "
            f"Because it could not be indexed, I cannot answer questions from it yet.{ready_context} "
            "Please upload a valid, text-searchable PDF."
        )

    # 2. Check if user mentions a specific failed document by name
    for f_doc in failed_docs:
        f_name = f_doc.get("filename", "")
        f_stem = Path(f_name).stem.lower().replace("_", " ").replace("-", " ")
        f_keywords = [w for w in f_stem.split() if len(w) > 2 and w not in ("technical", "documentation", "project", "automation", "pdf")]
        if f_keywords and any(kw in lower_query for kw in f_keywords):
            f_reason = f_doc.get("error", "Processing failed.")
            return (
                f"The document '{f_name}' could not be processed: {f_reason} "
                "Because text could not be extracted or indexed, I cannot answer questions from its content. "
                "Please upload a clean copy of this document."
            )

    # 3. If we have ready documents, perform retrieval
    if ready_docs:
        target_doc_id = None
        target_doc_name = None

        # If user explicitly asked for the "new pdf" / "latest pdf", target the most recent ready document
        if is_asking_for_new_doc:
            target_doc = ready_docs[-1]
            target_doc_id = target_doc.get("doc_id")
            target_doc_name = target_doc.get("filename")

        # Check if query matches a specific ready document name
        if not target_doc_id:
            for doc in ready_docs:
                fname = doc.get("filename", "")
                stem = Path(fname).stem.lower().replace("_", " ").replace("-", " ")
                keywords = [w for w in stem.split() if len(w) > 2 and w not in ("technical", "documentation", "project", "automation", "pdf")]
                if keywords and any(kw in lower_query for kw in keywords):
                    target_doc_id = doc.get("doc_id")
                    target_doc_name = doc.get("filename")
                    break

        retriever = get_retriever(thread_id, target_doc_id=target_doc_id)
        if retriever is None:
            return (
                "A PDF is registered for this chat, but its index could not be "
                "loaded. Please upload the PDF again."
            )

        index = retriever.vectorstore
        k = 4 if target_doc_id else min(12, max(4, 3 * len(ready_docs)))
        documents = retrieve_relevant_documents(index, query, k=k, score_threshold=0.06)

        # If strict threshold returned nothing, retry with a more permissive threshold to avoid empty answers
        if not documents:
            documents = retrieve_relevant_documents(index, query, k=k, score_threshold=0.01)

        if not documents:
            return (
                "No relevant information was found in the indexed PDF(s) "
                "uploaded to this chat for your query. The document content "
                "does not appear to support an answer to this question."
            )

        formatted = format_retrieved_documents(documents)
        if len(ready_docs) > 1 and not target_doc_id:
            active_list = ", ".join(d.get("filename", "Doc") for d in ready_docs)
            return f"[Active Documents in this conversation: {active_list}]\n\n{formatted}"
        return formatted

    # 4. If no ready docs, but failed docs exist
    if failed_docs:
        latest_failed = failed_docs[-1]
        filename = latest_failed.get("filename", "the uploaded document")
        failure_reason = latest_failed.get("error", "Text extraction produced 0 characters.")
        return (
            f"A PDF ('{filename}') was uploaded to this conversation, but processing failed: {failure_reason} "
            "Because text could not be extracted or indexed, I cannot answer questions from its content yet. "
            "Please upload an unlocked, text-searchable PDF (or use a document with selectable text), and try again."
        )

    # 5. No document (neither ready nor failed) has been uploaded in this thread
    return (
        "I can't answer this from a PDF yet because no PDF has been "
        "uploaded and indexed in this chat. Please attach the relevant "
        "PDF using the + button in the chat input, wait for the upload "
        "confirmation, and ask your question again. You can still ask "
        "general questions, request web searches, or ask about weather "
        "without uploading a PDF."
    )


# Tool list & LLM binding
tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool,
    purchase_stock,
]

llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


def chat_node(state: ChatState, config: RunnableConfig) -> Dict[str, Any]:
    """Use deterministic routing for clear tool intents, then let the LLM answer."""

    latest_message = state["messages"][-1] if state.get("messages") else None
    if isinstance(latest_message, HumanMessage):
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        previous_rag_call = any(
            getattr(message, "name", None) == "rag_tool"
            or any(
                tool_call.get("name") == "rag_tool"
                for tool_call in (
                    getattr(message, "tool_calls", None) or []
                )
                if isinstance(tool_call, dict)
            )
            for message in state.get("messages", [])[:-1]
        )
        has_thread_docs = bool(thread_id) and (
            has_rag_index(thread_id) or bool(_read_document_manifest(thread_id))
        )
        document_context = previous_rag_call or has_thread_docs
        forced_tool_call = _get_deterministic_tool_call(
            str(latest_message.content),
            document_context=document_context,
        )
        if forced_tool_call is not None:
            return {"messages": [AIMessage(content="", tool_calls=[forced_tool_call])]}

    system_message = get_system_message()
    messages = [
        system_message,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Postgres Saver Checkpointer
checkpoint = get_checkpointer()

# StateGraph construction
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads() -> List[str]:
    """Return all conversation thread IDs."""
    return service_get_all_threads(checkpoint)


__all__ = [
    "graph",
    "checkpoint",
    "chatbot",
    "chat_node",
    "tool_node",
    "tools",
    "llm_with_tools",
    "tools_condition",
    "ChatState",
    "add_messages",
    "START",
    "END",
    "interrupt",
    "Command",
    "FAISS",
    "search_tool",
    "calculator",
    "get_stock_price",
    "get_current_weather",
    "rag_tool",
    "purchase_stock",
    "get_all_tools",
    "PdfIngestionError",
    "ingest_rag_document",
    "get_rag_documents",
    "has_rag_index",
    "has_thread_documents",
    "clear_failed_documents",
    "get_retriever",
    "check_ocr_availability",
    "check_ollama_health",
    "get_effective_ollama_config",
    "postgres_conn",
    "get_all_threads",
    "RAG_DATA_DIR",
    "OLLAMA_RAW_URL",
    "OLLAMA_MODEL_NAME",
    "OLLAMA_EFFECTIVE_URL",
    "llm",
    "embeddings",
    "_normalize_ollama_url",
    "_rag_db_path",
    "_manifest_path",
    "_is_faiss_index",
    "_read_document_manifest",
    "_write_document_manifest",
    "_extract_pdf_documents",
    "_get_deterministic_tool_call",
]
