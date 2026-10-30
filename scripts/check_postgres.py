"""
Diagnostic script to check PostgreSQL connection and LangGraph checkpoints.
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from app.config import settings
from database.postgres import get_postgres_connection
from database.checkpointing import get_postgres_checkpointer, list_all_checkpoint_threads


def main():
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("========================================")
    print("POSTGRESQL DATABASE DIAGNOSTICS")
    print("========================================")

    print(f"Host:     {settings.postgres_host}")
    print(f"Port:     {settings.postgres_port}")
    print(f"Database: {settings.postgres_db}")
    print(f"User:     {settings.postgres_user}")

    print("\nAttempting database connection...")
    try:
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
        print(f"[PASS] Connected successfully!")
        print(f"PostgreSQL version: {version}")
    except Exception as e:
        print(f"[FAIL] Could not connect to PostgreSQL: {e}")
        sys.exit(1)

    print("\nChecking LangGraph checkpoints...")
    try:
        saver = get_postgres_checkpointer(conn)
        threads = list_all_checkpoint_threads(saver)
        print(f"[PASS] Checkpoints accessible. Total conversation threads: {len(threads)}")
        if threads:
            print("Existing threads:")
            for t in threads[:10]:
                print(f"  - {t}")
            if len(threads) > 10:
                print(f"  ... and {len(threads) - 10} more")
    except Exception as e:
        print(f"[FAIL] Failed to inspect checkpoints: {e}")
        sys.exit(1)

    print("\nAll PostgreSQL diagnostic checks passed!")


if __name__ == "__main__":
    main()

