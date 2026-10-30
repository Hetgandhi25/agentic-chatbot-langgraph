"""
LangGraph PostgreSQL checkpointer management using ConnectionPool.
Ensures thread-safe concurrent checkpointing and pipeline execution.
"""

import logging
from typing import Any, Optional, Set, List
from langgraph.checkpoint.postgres import PostgresSaver

from database.postgres import get_postgres_pool

logger = logging.getLogger(__name__)

_global_checkpoint: Optional[PostgresSaver] = None


def get_postgres_checkpointer(
    pool_or_conn: Optional[Any] = None,
    setup_tables: bool = True,
    new_instance: bool = False,
) -> PostgresSaver:
    """Initialize or return the PostgresSaver checkpointer backed by ConnectionPool.
    
    If setup_tables is True, invokes checkpoint.setup() to ensure tables are ready.
    """
    global _global_checkpoint
    if not new_instance and _global_checkpoint is not None:
        return _global_checkpoint

    if pool_or_conn is None:
        pool_or_conn = get_postgres_pool()

    saver = PostgresSaver(pool_or_conn)
    if setup_tables:
        try:
            saver.setup()
        except Exception as e:
            logger.warning(f"Checkpointer setup encountered an exception: {e}")

    if not new_instance:
        _global_checkpoint = saver
    return saver


def list_all_checkpoint_threads(checkpointer: Optional[PostgresSaver] = None) -> List[str]:
    """Retrieve all unique conversation thread IDs from PostgreSQL checkpoints."""
    threads: Set[str] = set()
    try:
        pool = get_postgres_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT thread_id FROM checkpoints;")
                rows = cur.fetchall()
                for r in rows:
                    if r[0]:
                        threads.add(str(r[0]))
        return sorted(list(threads))
    except Exception as e:
        logger.warning(f"Direct query for distinct threads failed ({e}); falling back to checkpointer.list()")

    if checkpointer is None:
        checkpointer = get_postgres_checkpointer()
    try:
        for ckpt in checkpointer.list(None):
            cfg = ckpt.config
            if isinstance(cfg, dict) and "configurable" in cfg:
                thread_id = cfg["configurable"].get("thread_id")
                if thread_id:
                    threads.add(str(thread_id))
    except Exception as e:
        logger.error(f"Error listing checkpoint threads: {e}")
    return sorted(list(threads))
