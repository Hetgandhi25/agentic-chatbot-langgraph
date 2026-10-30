import os
import sys
import uuid
import shutil
import tempfile
from pathlib import Path

# Ensure UTF-8 output encoding
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from agentic_chatbot_hitl_backend_postgres import (
    chatbot,
    embeddings,
    get_rag_documents,
    has_rag_index,
    has_thread_documents,
    ingest_rag_document,
    check_ocr_availability,
    check_ollama_health,
    rag_tool,
    _extract_pdf_documents,
    _get_deterministic_tool_call,
    chat_node,
    PdfIngestionError,
    RAG_DATA_DIR,
)
from langchain_core.messages import HumanMessage, AIMessage


def run_diagnostics():
    print("==================================================")
    print("RUNNING CSM & STATE DIAGNOSTICS SUITE (WITH OCR)")
    print("==================================================")

    csm_pdf_path = Path(r"C:\Users\HET\Downloads\Csm Ppt Automation Documentation.pdf")
    het_pdf_path = Path(r"C:\Users\HET\Downloads\Het_Gandhi.pdf")

    # 1. Valid text-based PDF extracts and indexes successfully
    print("\n--- Test 1: Valid text-based PDF ingestion ---")
    thread_valid = f"test-valid-{uuid.uuid4()}"
    res_valid = ingest_rag_document(str(het_pdf_path), thread_valid, filename="Het Gandhi.pdf")
    assert res_valid["status"] == "ready", "Valid PDF must be marked as ready"
    assert has_rag_index(thread_valid), "Thread must have a valid RAG index"
    ready_docs = get_rag_documents(thread_valid, status="ready")
    assert len(ready_docs) == 1, "Expected 1 ready document"
    print("[PASS] Test 1: Valid text-based PDF extracted and indexed successfully.")

    # 2. OCR Availability Check
    print("\n--- Test 2: OCR availability check ---")
    ocr_available, ocr_msg = check_ocr_availability()
    print(f"OCR Availability: {ocr_available} ({ocr_msg})")
    assert ocr_available, "OCR must be available after Tesseract installation!"
    print("[PASS] Test 2: OCR engine detected and active.")

    # 3. Target CSM PDF ingestion with OCR
    print("\n--- Test 3: Target CSM PDF ingestion via OCR fallback ---")
    assert csm_pdf_path.is_file(), f"Target CSM PDF not found at {csm_pdf_path}"
    thread_csm = f"test-csm-{uuid.uuid4()}"
    res_csm = ingest_rag_document(
        str(csm_pdf_path),
        thread_csm,
        filename="Csm Ppt Automation Documentation.pdf",
    )
    print(f"CSM Ingestion Result: pages={res_csm['page_count']}, chars={res_csm['character_count']}, method={res_csm.get('extraction_method')}")
    assert res_csm["status"] == "ready", "CSM PDF must be marked as ready after OCR!"
    assert res_csm["page_count"] == 7, "CSM PDF must have 7 pages"
    assert res_csm["character_count"] > 5000, "CSM PDF must have extracted > 5000 characters via OCR"
    assert res_csm.get("extraction_method") == "ocr", "Extraction method should be OCR"
    assert has_rag_index(thread_csm), "CSM thread must have a ready RAG index"
    print("[PASS] Test 3: Target CSM PDF successfully rendered, OCR-transcribed, and indexed.")

    # 4. Chatbot response to 'explain this pdf in short' for CSM PDF
    print("\n--- Test 4: Chatbot response to 'explain this pdf in short' on CSM PDF ---")
    config_csm = {"configurable": {"thread_id": thread_csm}}
    rag_tool_output = rag_tool.func("explain this pdf in short", config_csm)
    print(f"RAG Tool Response Preview:\n  {rag_tool_output[:300]}...")
    assert "Csm Ppt Automation Documentation.pdf" in rag_tool_output
    assert "Ishan Analytics Hub" in rag_tool_output or "CSM" in rag_tool_output or "Automation" in rag_tool_output
    print("[PASS] Test 4: RAG tool retrieves real document content from CSM PDF.")

    # 5. Truly blank/invalid PDF is rejected and persistently recorded as Failed
    print("\n--- Test 5: Truly blank PDF rejection & Failed status tracking ---")
    thread_blank = f"test-blank-{uuid.uuid4()}"
    blank_pdf_path = Path(tempfile.gettempdir()) / f"blank_{uuid.uuid4()}.pdf"
    # Create empty 0-byte file
    blank_pdf_path.write_bytes(b"")
    try:
        ingest_rag_document(str(blank_pdf_path), thread_blank, filename="blank.pdf")
        assert False, "Empty PDF must raise error!"
    except PdfIngestionError as error:
        print(f"Captured expected error on empty PDF: {error}")
    finally:
        if blank_pdf_path.exists():
            blank_pdf_path.unlink()

    assert not has_rag_index(thread_blank), "Failed PDF must NOT have a ready RAG index!"
    failed_in_blank = get_rag_documents(thread_blank, status="failed")
    assert len(failed_in_blank) == 1, "Failed documents list must record the failed upload"
    assert failed_in_blank[0]["status"] == "failed"
    print("[PASS] Test 5: Invalid/empty PDF rejected and persistently recorded as Failed.")

    # 6. Chatbot response when user asks about a failed PDF
    print("\n--- Test 6: Chatbot response after PDF processing failure ---")
    config_blank = {"configurable": {"thread_id": thread_blank}}
    failed_tool_output = rag_tool.func("explain this pdf in short", config_blank)
    print(f"RAG Tool Response for Failed PDF:\n  {failed_tool_output}")
    assert "processing failed" in failed_tool_output.lower()
    print("[PASS] Test 6: Chatbot accurately explains why the PDF failed instead of claiming no PDF exists.")

    # 7. Chatbot response in a fresh thread with no PDF
    print("\n--- Test 7: Chatbot response in a fresh thread with no PDF ---")
    thread_fresh = f"test-fresh-{uuid.uuid4()}"
    config_fresh = {"configurable": {"thread_id": thread_fresh}}
    fresh_output = rag_tool.func("explain this pdf in short", config_fresh)
    print(f"RAG Tool Response for Empty Thread:\n  {fresh_output}")
    assert "no pdf has been uploaded and indexed in this chat" in fresh_output.lower()
    print("[PASS] Test 7: Chatbot prompts to attach a PDF only when truly no PDF was ever uploaded.")

    # 8. Multi-conversation document isolation
    print("\n--- Test 8: Multi-conversation document isolation ---")
    assert not has_rag_index(thread_fresh), "Fresh thread must NOT see documents from CSM thread!"
    csm_query_in_fresh = rag_tool.func("What is Ishan Analytics Hub?", config_fresh)
    assert "no pdf has been uploaded" in csm_query_in_fresh.lower()
    print("[PASS] Test 8: Thread isolation verified; documents are never leaked across conversations.")

    # 9. State persistence across lookups
    print("\n--- Test 9: State persistence across lookups ---")
    restored_csm = get_rag_documents(thread_csm, status="ready")
    assert len(restored_csm) == 1
    assert restored_csm[0]["filename"] == "Csm Ppt Automation Documentation.pdf"
    assert restored_csm[0]["status"] == "ready"
    assert restored_csm[0]["page_count"] == 7
    print("[PASS] Test 9: Document metadata persistently distinguishes Ready vs Failed.")

    print("\n==================================================")
    print("ALL CSM & STATE DIAGNOSTIC TESTS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    run_diagnostics()
