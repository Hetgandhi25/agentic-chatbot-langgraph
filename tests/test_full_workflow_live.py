import os
import sys
import uuid
import shutil
from pathlib import Path

# Ensure utf-8 output encoding on Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from agentic_chatbot_hitl_backend_postgres import (
    chatbot,
    embeddings,
    get_effective_ollama_config,
    check_ollama_health,
    _extract_pdf_documents,
    ingest_rag_document,
    get_rag_documents,
    has_rag_index,
    rag_tool,
    RAG_DATA_DIR,
)
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

def test_full_workflow():
    print("==================================================")
    print("STARTING FULL END-TO-END WORKFLOW VERIFICATION")
    print("==================================================")

    # 1. Ollama health check using the actual configured endpoint
    print("\n--- Test 1: Ollama health check ---")
    raw_url, model_name = get_effective_ollama_config()
    print(f"Configured raw URL: {raw_url}, model: {model_name}")
    health = check_ollama_health(timeout=5.0)
    print(f"Health check result: {health}")
    assert health["status"] == "healthy", "Ollama health check failed!"
    print("[PASS] Test 1: Ollama health check is healthy.")

    # 2. Embedding request using the actual configured model
    print("\n--- Test 2: Embedding request ---")
    query_text = "Senior Python and RAG engineer testing embeddings."
    vector = embeddings.embed_query(query_text)
    print(f"Generated embedding vector length: {len(vector)}")
    print(f"Sample vector elements: {vector[:5]}")
    assert len(vector) > 0, "Embedding vector is empty!"
    print("[PASS] Test 2: Actual model generated embeddings successfully.")

    # 3. PDF extraction without invoking Ollama
    print("\n--- Test 3: PDF extraction without invoking Ollama ---")
    real_pdf_path = Path(r"C:\Users\HET\Downloads\Het_Gandhi.pdf")
    assert real_pdf_path.is_file(), f"Real PDF file not found at {real_pdf_path}"
    temp_thread = f"test-extract-{uuid.uuid4()}"
    docs, page_count, char_count, errors = _extract_pdf_documents(
        str(real_pdf_path), temp_thread, "Het Gandhi.pdf"
    )
    print(f"Pages: {page_count}, Characters: {char_count}, Chunks extracted: {len(docs)}, Errors: {errors}")
    assert page_count >= 1, "Page count should be at least 1"
    assert char_count > 500, "Should have extracted text from the PDF"
    print("[PASS] Test 3: PDF extraction succeeded independently of Ollama.")

    # 4 & 5. Embedding generation and FAISS indexing with real file
    print("\n--- Test 4 & 5: FAISS Indexing with real Het_Gandhi.pdf ---")
    thread_a = f"thread-a-{uuid.uuid4()}"
    ingest_result = ingest_rag_document(
        str(real_pdf_path),
        thread_a,
        filename="Het Gandhi.pdf",
    )
    print(f"Ingestion result: {ingest_result}")
    assert ingest_result["status"] == "ready", "Status should be ready"
    assert has_rag_index(thread_a), "Thread A must have RAG index"
    rag_docs = get_rag_documents(thread_a)
    assert len(rag_docs) == 1, "Expected 1 registered document in thread A"
    assert rag_docs[0]["filename"] == "Het Gandhi.pdf"
    print("[PASS] Test 4 & 5: FAISS index and persistent metadata created for Het Gandhi.pdf.")

    # 6. Ask the chatbot to summarize the successfully indexed document
    print("\n--- Test 6: Chatbot summary of indexed document ---")
    config_a = {"configurable": {"thread_id": thread_a}}
    summary_input = "Please summarize the uploaded document Het Gandhi.pdf."
    summary_response = None
    for chunk, meta in chatbot.stream(
        {"messages": [HumanMessage(content=summary_input)]},
        config=config_a,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage) and chunk.content and not getattr(chunk, "tool_calls", None):
            summary_response = (summary_response or "") + chunk.content

    print(f"Chatbot Summary Response:\n{summary_response}")
    assert summary_response and len(summary_response) > 50, "Chatbot summary should not be empty!"
    print("[PASS] Test 6: Chatbot successfully summarized the indexed PDF.")

    # 7. Ask a specific follow-up question whose answer is present in the PDF
    print("\n--- Test 7: Specific follow-up question ---")
    followup_input = "What is Het Gandhi's CPI and what university does he attend according to the document?"
    followup_response = None
    for chunk, meta in chatbot.stream(
        {"messages": [HumanMessage(content=followup_input)]},
        config=config_a,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage) and chunk.content and not getattr(chunk, "tool_calls", None):
            followup_response = (followup_response or "") + chunk.content

    print(f"Chatbot Follow-up Response:\n{followup_response}")
    assert followup_response and ("6.78" in followup_response or "dhirubhai" in followup_response.lower() or "ambani" in followup_response.lower()), (
        f"Expected CPI 6.78 or Dhirubhai Ambani in answer, got: {followup_response}"
    )
    print("[PASS] Test 7: Chatbot answered specific factual question from the PDF correctly.")

    # 8. Verify document availability persists for thread A
    print("\n--- Test 8: Document persistence verification ---")
    assert has_rag_index(thread_a), "Thread A must retain its RAG index"
    restored_docs = get_rag_documents(thread_a)
    assert len(restored_docs) == 1 and restored_docs[0]["filename"] == "Het Gandhi.pdf"
    print("[PASS] Test 8: Document availability persists across lookups.")

    # 9. Switch to another conversation and verify document isolation
    print("\n--- Test 9: Document isolation between threads ---")
    thread_b = f"thread-b-{uuid.uuid4()}"
    config_b = {"configurable": {"thread_id": thread_b}}
    assert not has_rag_index(thread_b), "Thread B should NOT have any RAG index"
    assert len(get_rag_documents(thread_b)) == 0, "Thread B should have 0 registered documents"

    # Query in Thread B asking about the document
    rag_tool_b_output = rag_tool.func(
        "What is Het Gandhi CPI?",
        config_b,
    )
    print(f"Thread B RAG tool direct output: {rag_tool_b_output}")
    assert "no pdf has been uploaded" in rag_tool_b_output.lower(), "Thread B must not access Thread A's document!"
    print("[PASS] Test 9: Perfect thread isolation confirmed.")

    # 10. Regression tests for PostgreSQL persistence, streaming, and HITL
    print("\n--- Test 10: PostgreSQL persistence and HITL regression ---")
    thread_hitl = f"thread-hitl-{uuid.uuid4()}"
    config_hitl = {"configurable": {"thread_id": thread_hitl}}

    # Stream a HITL stock purchase request
    hitl_input = "Buy 10 shares of AAPL"
    interrupted = False
    for chunk, meta in chatbot.stream(
        {"messages": [HumanMessage(content=hitl_input)]},
        config=config_hitl,
        stream_mode="messages",
    ):
        pass

    state_snapshot = chatbot.get_state(config_hitl)
    interrupts = getattr(state_snapshot, "interrupts", ()) or ()
    if not interrupts:
        tasks = getattr(state_snapshot, "tasks", ()) or ()
        for t in tasks:
            if getattr(t, "interrupts", ()):
                interrupts = t.interrupts
                break

    print(f"Pending interrupts in PostgreSQL checkpoint: {interrupts}")
    assert len(interrupts) > 0, "Expected HITL interruption for stock purchase!"
    print(f"Interrupt prompt: {interrupts[0].value}")

    # Resume HITL execution with 'yes'
    resumed_response = None
    for chunk, meta in chatbot.stream(
        Command(resume="yes"),
        config=config_hitl,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage) and chunk.content and not getattr(chunk, "tool_calls", None):
            resumed_response = (resumed_response or "") + chunk.content

    print(f"Resumed response after human approval: {resumed_response}")
    post_state = chatbot.get_state(config_hitl)
    assert len(post_state.values.get("messages", [])) >= 3, "Messages should be saved in PostgreSQL checkpoint"
    print("[PASS] Test 10: PostgreSQL checkpointing, streaming, and HITL approval fully verified.")

    # Clean up test directories
    for tid in (thread_a, thread_b, thread_hitl):
        p = RAG_DATA_DIR / tid
        if p.exists():
            shutil.rmtree(p)

    print("\n==================================================")
    print("ALL 10 VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    test_full_workflow()
