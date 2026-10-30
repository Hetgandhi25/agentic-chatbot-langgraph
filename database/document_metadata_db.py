"""
PostgreSQL persistence repository for RAG document metadata using ConnectionPool.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

from database.postgres import get_postgres_pool

logger = logging.getLogger(__name__)

_table_initialized = False


def init_document_metadata_table() -> None:
    """Initialize the document_metadata table and indices in PostgreSQL if not present."""
    global _table_initialized
    if _table_initialized:
        return

    try:
        pool = get_postgres_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    id SERIAL PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    uploaded_at TIMESTAMPTZ NOT NULL,
                    page_count INTEGER,
                    character_count INTEGER,
                    chunk_count INTEGER,
                    index_path TEXT,
                    extraction_method TEXT,
                    diagnostic TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_meta_thread_id ON document_metadata(thread_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_meta_status ON document_metadata(status);")
        _table_initialized = True
        logger.info("document_metadata table initialized in PostgreSQL.")
    except Exception as e:
        logger.warning(f"Could not initialize document_metadata table in PostgreSQL: {e}")


def upsert_document_metadata_to_db(thread_id: str, doc: Dict[str, Any]) -> bool:
    """Insert or update a document metadata record in PostgreSQL."""
    init_document_metadata_table()
    upsert_sql = """
    INSERT INTO document_metadata (
        thread_id, doc_id, filename, status, error, uploaded_at,
        page_count, character_count, chunk_count, index_path,
        extraction_method, diagnostic, updated_at
    ) VALUES (
        %(thread_id)s, %(doc_id)s, %(filename)s, %(status)s, %(error)s, %(uploaded_at)s,
        %(page_count)s, %(character_count)s, %(chunk_count)s, %(index_path)s,
        %(extraction_method)s, %(diagnostic)s, NOW()
    )
    ON CONFLICT (doc_id) DO UPDATE SET
        filename = EXCLUDED.filename,
        status = EXCLUDED.status,
        error = EXCLUDED.error,
        uploaded_at = EXCLUDED.uploaded_at,
        page_count = EXCLUDED.page_count,
        character_count = EXCLUDED.character_count,
        chunk_count = EXCLUDED.chunk_count,
        index_path = EXCLUDED.index_path,
        extraction_method = EXCLUDED.extraction_method,
        diagnostic = EXCLUDED.diagnostic,
        updated_at = NOW();
    """
    uploaded_at = doc.get("uploaded_at")
    if isinstance(uploaded_at, str):
        try:
            uploaded_at_dt = datetime.fromisoformat(uploaded_at)
        except Exception:
            uploaded_at_dt = datetime.now(timezone.utc)
    elif isinstance(uploaded_at, datetime):
        uploaded_at_dt = uploaded_at
    else:
        uploaded_at_dt = datetime.now(timezone.utc)

    def clean_str(val: Any) -> Optional[str]:
        if val is None:
            return None
        s = str(val).replace("\x00", "")
        try:
            s = s.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
        except Exception:
            pass
        return s.encode("utf-8", "ignore").decode("utf-8", "ignore").replace("\x00", "")

    params = {
        "thread_id": clean_str(thread_id),
        "doc_id": clean_str(str(doc.get("doc_id"))),
        "filename": clean_str(doc.get("filename", "unknown.pdf")),
        "status": clean_str(doc.get("status", "ready")),
        "error": clean_str(doc.get("error")),
        "uploaded_at": uploaded_at_dt,
        "page_count": doc.get("page_count"),
        "character_count": doc.get("character_count", 0),
        "chunk_count": doc.get("chunk_count", 0),
        "index_path": clean_str(doc.get("index_path")),
        "extraction_method": clean_str(doc.get("extraction_method", "native")),
        "diagnostic": clean_str(doc.get("diagnostic")),
    }
    try:
        pool = get_postgres_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(upsert_sql, params)
        return True
    except Exception as e:
        logger.warning(f"Failed to upsert document metadata to PostgreSQL for {doc.get('doc_id')}: {e}")
        return False


def get_thread_documents_from_db(thread_id: str, status: Optional[str] = "ready") -> List[Dict[str, Any]]:
    """Retrieve document records from PostgreSQL for a conversation thread."""
    init_document_metadata_table()
    if status is not None and status != "all":
        query_sql = """
        SELECT doc_id, filename, status, error, uploaded_at, page_count,
               character_count, chunk_count, index_path, extraction_method, diagnostic
        FROM document_metadata
        WHERE thread_id = %(thread_id)s AND status = %(status)s
        ORDER BY uploaded_at ASC;
        """
        params = {"thread_id": thread_id, "status": status}
    else:
        query_sql = """
        SELECT doc_id, filename, status, error, uploaded_at, page_count,
               character_count, chunk_count, index_path, extraction_method, diagnostic
        FROM document_metadata
        WHERE thread_id = %(thread_id)s
        ORDER BY uploaded_at ASC;
        """
        params = {"thread_id": thread_id}

    try:
        pool = get_postgres_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_sql, params)
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "doc_id": row[0],
                        "filename": row[1],
                        "status": row[2],
                        "error": row[3],
                        "uploaded_at": row[4].isoformat() if row[4] else None,
                        "page_count": row[5],
                        "character_count": row[6],
                        "chunk_count": row[7],
                        "index_path": row[8],
                        "extraction_method": row[9],
                        "diagnostic": row[10],
                    })
                return results
    except Exception as e:
        logger.warning(f"Could not query PostgreSQL for thread documents ({thread_id}): {e}")
        return []


def delete_document_from_db(thread_id: str, doc_id: str) -> bool:
    """Delete a document record from PostgreSQL."""
    init_document_metadata_table()
    try:
        pool = get_postgres_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_metadata WHERE thread_id = %s AND doc_id = %s;",
                    (thread_id, doc_id),
                )
        return True
    except Exception as e:
        logger.warning(f"Failed to delete document {doc_id} from PostgreSQL: {e}")
        return False


def clear_failed_documents_from_db(thread_id: str) -> bool:
    """Remove all failed document records for a thread from PostgreSQL."""
    init_document_metadata_table()
    try:
        pool = get_postgres_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_metadata WHERE thread_id = %s AND status = 'failed';",
                    (thread_id,),
                )
        return True
    except Exception as e:
        logger.warning(f"Failed to clear failed documents from PostgreSQL for thread {thread_id}: {e}")
        return False


def delete_failed_documents_by_filename(thread_id: str, filename: str) -> bool:
    """Remove failed document records for a specific filename in a thread when it succeeds."""
    init_document_metadata_table()
    try:
        pool = get_postgres_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_metadata WHERE thread_id = %s AND filename = %s AND status = 'failed';",
                    (thread_id, filename),
                )
        return True
    except Exception as e:
        logger.warning(f"Failed to delete failed records for {filename} from PostgreSQL: {e}")
        return False

