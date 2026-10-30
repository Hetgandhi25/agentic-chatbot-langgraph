"""
Core application package containing configuration, logging, and dependencies.
"""

from app.config import settings
from app.logging_config import setup_logging, get_logger

__all__ = [
    "settings",
    "setup_logging",
    "get_logger",
    "get_llm",
    "get_embeddings",
    "get_db_connection",
    "get_checkpointer",
]


def __getattr__(name: str):
    if name in {"get_llm", "get_embeddings", "get_db_connection", "get_checkpointer"}:
        from app import dependencies
        return getattr(dependencies, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


