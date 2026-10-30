"""
PostgreSQL database connection utilities using psycopg and psycopg_pool.
Provides thread-safe ConnectionPool for LangGraph checkpoints and concurrent queries.
"""

import logging
from typing import Optional
import psycopg
from psycopg import Connection
from psycopg_pool import ConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

_global_pool: Optional[ConnectionPool] = None


def get_postgres_pool() -> ConnectionPool:
    """Return the shared thread-safe ConnectionPool for PostgreSQL."""
    global _global_pool
    if _global_pool is None or _global_pool.closed:
        logger.info(
            f"Initializing PostgreSQL ConnectionPool for {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
        _global_pool = ConnectionPool(
            conninfo="",
            min_size=2,
            max_size=20,
            kwargs={
                "host": settings.postgres_host,
                "port": settings.postgres_port,
                "dbname": settings.postgres_db,
                "user": settings.postgres_user,
                "password": settings.postgres_password,
                "autocommit": True,
                "prepare_threshold": 0,
            },
            open=True,
        )
    return _global_pool


def get_postgres_connection(new_instance: bool = False) -> Connection:
    """Create or return a standalone PostgreSQL database connection."""
    logger.info(
        f"Connecting to PostgreSQL at {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db} "
        f"as user {settings.postgres_user}"
    )
    return psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        autocommit=True,
        prepare_threshold=0,
    )
