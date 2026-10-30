"""
Conversation and checkpoint management service.
"""

from typing import List, Optional, Set
from langgraph.checkpoint.postgres import PostgresSaver

from app.dependencies import get_checkpointer


def get_all_threads(checkpointer: Optional[PostgresSaver] = None) -> List[str]:
    """Retrieve all unique conversation thread IDs from PostgreSQL checkpoints."""
    saver = checkpointer or get_checkpointer()
    all_threads: Set[str] = set()

    try:
        for ckpt in saver.list(None):
            cfg = getattr(ckpt, "config", None)
            if isinstance(cfg, dict) and "configurable" in cfg:
                thread_id = cfg["configurable"].get("thread_id")
                if thread_id:
                    all_threads.add(str(thread_id))
    except Exception:
        pass

    return list(all_threads)

