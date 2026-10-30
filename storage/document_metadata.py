"""
Document metadata persistence and manifest management for RAG documents.
Provides dual persistence: PostgreSQL document_metadata table + local documents.json.
Includes Windows-safe file and directory operations with lock retry.
"""

from datetime import datetime, timezone
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

from storage.paths import RAG_DATA_DIR, get_manifest_path, get_rag_db_path, sanitize_thread_id
from database.document_metadata_db import (
    init_document_metadata_table,
    upsert_document_metadata_to_db,
    get_thread_documents_from_db,
    clear_failed_documents_from_db,
    delete_failed_documents_by_filename,
)

logger = logging.getLogger(__name__)


def safe_remove_directory(
    target_path: Path,
    max_retries: int = 5,
    delay: float = 0.1,
) -> None:
    """Safely remove a directory tree with retries on Windows for transient locks."""
    if not target_path.exists():
        return
    gc.collect()
    for attempt in range(max_retries):
        try:
            shutil.rmtree(target_path, ignore_errors=False)
            return
        except (PermissionError, OSError):
            if attempt == max_retries - 1:
                shutil.rmtree(target_path, ignore_errors=True)
                return
            time.sleep(delay * (1.2 ** attempt))
            gc.collect()


def safe_publish_directory(
    staged_path: Path,
    final_path: Path,
    max_retries: int = 10,
    delay: float = 0.1,
) -> None:
    """Safely move or publish a staged FAISS index directory to final_path on Windows.
    
    Handles Windows file-locking, antivirus scanners, and directory rename restrictions:
      1. Performs garbage collection to release lingering Python-level handles.
      2. If final_path exists, cleans it up safely before moving.
      3. Attempts atomic directory move/replace (os.replace) with retry and backoff.
      4. If directory-level rename is locked by external scanners, falls back to
         atomic file-by-file copy into final_path followed by staged_path cleanup.
    """
    if not staged_path.exists():
        raise FileNotFoundError(f"Staged index path does not exist: {staged_path}")

    gc.collect()

    if final_path.exists():
        safe_remove_directory(final_path, max_retries=max_retries, delay=delay)

    final_path.parent.mkdir(parents=True, exist_ok=True)

    last_err = None
    for attempt in range(max_retries):
        try:
            os.replace(staged_path, final_path)
            return
        except (PermissionError, OSError) as e:
            last_err = e
            time.sleep(delay * (1.2 ** attempt))
            gc.collect()

    # Fallback: copy files individually into final_path
    try:
        final_path.mkdir(parents=True, exist_ok=True)
        for item in staged_path.iterdir():
            target_file = final_path / item.name
            for attempt in range(max_retries):
                try:
                    if item.is_dir():
                        shutil.copytree(item, target_file, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target_file)
                    break
                except (PermissionError, OSError) as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (1.2 ** attempt))
                    gc.collect()

        safe_remove_directory(staged_path, max_retries=max_retries, delay=delay)
    except Exception as copy_err:
        raise PermissionError(
            f"Failed to publish index from '{staged_path}' to '{final_path}' after "
            f"{max_retries} retries due to file lock or permissions. "
            f"Original error: {last_err}. Fallback error: {copy_err}"
        ) from copy_err


def read_document_manifest(thread_id: str, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read document manifest for a conversation thread.
    
    Reads from PostgreSQL as primary source of truth, falls back to documents.json,
    and automatically synchronizes both stores.
    """
    safe_tid = sanitize_thread_id(thread_id)
    manifest_path = get_manifest_path(safe_tid, base_dir=base_dir)
    is_custom_dir = base_dir is not None and Path(base_dir).resolve() != RAG_DATA_DIR.resolve()

    # 1. Read local JSON manifest if present
    json_docs: List[Dict[str, Any]] = []
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "documents" in data and isinstance(data["documents"], list):
                    json_docs = data["documents"]
                elif isinstance(data, list):
                    json_docs = data
        except Exception as e:
            logger.warning(f"Could not read local document manifest at {manifest_path}: {e}")

    # If isolated to a custom base_dir (such as a tempdir in automated tests), return local JSON
    if is_custom_dir:
        return json_docs

    # 2. Read PostgreSQL records
    db_docs: List[Dict[str, Any]] = []
    try:
        db_docs = get_thread_documents_from_db(safe_tid, status=None)
    except Exception as e:
        logger.warning(f"Could not query PostgreSQL for thread {safe_tid}: {e}")


    # 3. Synchronize: If PostgreSQL has docs, use them and ensure JSON is up to date
    if db_docs:
        # Merge any unique doc_id from json_docs into db if missing in db
        db_doc_ids = {d.get("doc_id") for d in db_docs if d.get("doc_id")}
        for j_doc in json_docs:
            j_id = j_doc.get("doc_id")
            if j_id and j_id not in db_doc_ids:
                try:
                    upsert_document_metadata_to_db(safe_tid, j_doc)
                    db_docs.append(j_doc)
                except Exception:
                    pass
        return db_docs

    # 4. If PostgreSQL has no docs but JSON does, backfill to PostgreSQL
    if json_docs:
        for j_doc in json_docs:
            if j_doc.get("doc_id"):
                try:
                    upsert_document_metadata_to_db(safe_tid, j_doc)
                except Exception:
                    pass
        return json_docs

    return []


def write_document_manifest(
    thread_id: str,
    docs: List[Dict[str, Any]],
    base_dir: Optional[Path] = None,
) -> None:
    """Atomically write document manifest for a conversation thread to both PostgreSQL and JSON."""
    safe_tid = sanitize_thread_id(thread_id)
    is_custom_dir = base_dir is not None and Path(base_dir).resolve() != RAG_DATA_DIR.resolve()

    # 1. Persist to PostgreSQL if in production database directory
    if not is_custom_dir:
        for doc in docs:
            if doc.get("doc_id"):
                try:
                    if doc.get("status") == "ready" and doc.get("filename"):
                        delete_failed_documents_by_filename(safe_tid, doc["filename"])
                    upsert_document_metadata_to_db(safe_tid, doc)
                except Exception as e:
                    logger.warning(f"Could not persist doc {doc.get('doc_id')} to PostgreSQL: {e}")

    # 2. Atomically write to local documents.json
    thread_path = get_rag_db_path(safe_tid, base_dir=base_dir)
    thread_path.mkdir(parents=True, exist_ok=True)
    manifest_path = get_manifest_path(safe_tid, base_dir=base_dir)
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
            json.dump({"documents": docs}, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        # Safe replace with retry loop for Windows locks
        for attempt in range(10):
            try:
                os.replace(temporary_path, manifest_path)
                temporary_path = None
                break
            except (PermissionError, OSError):
                time.sleep(0.05 * (attempt + 1))
        if temporary_path is not None and temporary_path.exists():
            shutil.copy2(temporary_path, manifest_path)
            try:
                temporary_path.unlink()
            except OSError:
                pass
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def record_document_status(
    thread_id: str,
    doc_id: str,
    filename: str,
    status: str,
    page_count: Optional[int] = None,
    character_count: Optional[int] = None,
    chunk_count: Optional[int] = None,
    error_message: Optional[str] = None,
    index_path: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Record or update a document's status in the thread manifest."""
    docs = read_document_manifest(thread_id, base_dir=base_dir)
    now = datetime.now(timezone.utc).isoformat()
    
    entry = None
    for item in docs:
        if item.get("doc_id") == doc_id:
            entry = item
            break
            
    if entry is None:
        entry = {
            "doc_id": doc_id,
            "filename": filename,
            "created_at": now,
        }
        docs.append(entry)
        
    entry["status"] = status
    entry["updated_at"] = now
    if page_count is not None:
        entry["page_count"] = page_count
    if character_count is not None:
        entry["character_count"] = character_count
    if chunk_count is not None:
        entry["chunk_count"] = chunk_count
    if index_path is not None:
        entry["index_path"] = index_path
    if error_message is not None:
        entry["error"] = error_message
        entry["error_message"] = error_message
    elif status.lower() == "ready":
        entry.pop("error", None)
        entry.pop("error_message", None)
        
    if extra_metadata:
        entry.update(extra_metadata)
        
    write_document_manifest(thread_id, docs, base_dir=base_dir)
    return entry
