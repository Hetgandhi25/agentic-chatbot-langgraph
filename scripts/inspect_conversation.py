"""
Inspect conversation checkpoint state and message history for a given thread_id.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from agent import chatbot
from services.conversation_service import get_all_threads
from services.document_service import get_rag_documents


def main():
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Inspect a conversation thread in PostgreSQL checkpoints.")
    parser.add_argument("--thread-id", "-t", type=str, default=None, help="The conversation thread ID to inspect.")
    parser.add_argument("--list", "-l", action="store_true", help="List all known thread IDs.")
    args = parser.parse_args()

    threads = get_all_threads()

    if args.list or not args.thread_id:
        print("========================================")
        print(f"AVAILABLE CONVERSATION THREADS ({len(threads)})")
        print("========================================")
        for t in threads:
            docs = get_rag_documents(t, status="all")
            doc_info = f" [PDFs: {len(docs)}]" if docs else ""
            print(f"- {t}{doc_info}")
        if not args.thread_id:
            print("\nUsage: python scripts/inspect_conversation.py --thread-id <THREAD_ID>")
            return

    thread_id = args.thread_id
    print("========================================")
    print(f"INSPECTING THREAD: {thread_id}")
    print("========================================")

    docs = get_rag_documents(thread_id, status="all")
    if docs:
        print("\nRegistered Documents:")
        for d in docs:
            print(f"  - {d.get('filename')} (status: {d.get('status')}, pages: {d.get('page_count')})")
    else:
        print("\nNo registered documents found for this thread.")

    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    if not state or not state.values:
        print(f"\nNo checkpoint state found for thread '{thread_id}'.")
        return

    messages = state.values.get("messages", [])
    print(f"\nTotal messages in checkpoint: {len(messages)}")
    for i, msg in enumerate(messages, start=1):
        role = type(msg).__name__
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", None)
        print(f"\n[{i}] {role}:")
        if content:
            print(f"    {content}")
        if tool_calls:
            print(f"    Tool calls: {tool_calls}")

    if state.next:
        print(f"\nPending next nodes (interrupt): {state.next}")


if __name__ == "__main__":
    main()

