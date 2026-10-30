"""
Regression test suite for multi-PDF uploads in the same conversation,
cross-thread isolation, duplicate filenames, failed ingestion cleanup,
and PostgreSQL document metadata persistence.
"""

import os
import sys
from pathlib import Path
import shutil
import unittest
import uuid

# Ensure utf-8 output encoding on Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from services.pdf_ingestion_service import ingest_rag_document
from services.document_service import get_rag_documents, get_retriever, has_rag_index
from tools.rag.tool import rag_tool
from database.document_metadata_db import get_thread_documents_from_db
from storage.paths import RAG_DATA_DIR, is_faiss_index
from tools.pdf import PdfIngestionError


class TestMultiPdfUpload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.debate_pdf = Path(r"C:\Users\HET\Downloads\AI Debate Agent Notes.pdf")
        cls.mbr_pdf = Path(r"C:\Users\HET\Downloads\MBR_Automation_Project_Documentation.pdf")
        cls.csm_pdf = Path(r"C:\Users\HET\Downloads\Csm Ppt Automation Documentation.pdf")
        
        assert cls.debate_pdf.is_file(), f"Missing test PDF at {cls.debate_pdf}"
        assert cls.mbr_pdf.is_file(), f"Missing test PDF at {cls.mbr_pdf}"

    def setUp(self):
        self.created_threads = []

    def tearDown(self):
        for tid in self.created_threads:
            p = RAG_DATA_DIR / tid
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

    def test_sequential_multi_pdf_upload_in_same_thread(self):
        """Test uploading two distinct PDFs sequentially in the exact same conversation thread.
        Verifies that no [WinError 5] occurs and both documents remain registered and queryable.
        """
        thread_id = f"test-multi-{uuid.uuid4()}"
        self.created_threads.append(thread_id)

        # 1. Ingest first PDF (AI Debate Agent Notes)
        res1 = ingest_rag_document(
            str(self.debate_pdf),
            thread_id,
            filename="AI Debate Agent Notes.pdf",
        )
        self.assertEqual(res1["status"], "ready")
        doc_id_1 = res1["doc_id"]
        self.assertTrue(doc_id_1)
        self.assertTrue(is_faiss_index(RAG_DATA_DIR / res1["index_path"]))

        # 2. Ingest second PDF (MBR Automation Project Documentation) in the SAME thread
        res2 = ingest_rag_document(
            str(self.mbr_pdf),
            thread_id,
            filename="MBR_Automation_Project_Documentation.pdf",
        )
        self.assertEqual(res2["status"], "ready")
        doc_id_2 = res2["doc_id"]
        self.assertTrue(doc_id_2)
        self.assertNotEqual(doc_id_1, doc_id_2)
        self.assertTrue(is_faiss_index(RAG_DATA_DIR / res2["index_path"]))

        # 3. Verify manifest contains BOTH documents
        docs = get_rag_documents(thread_id, status="ready")
        self.assertEqual(len(docs), 2, "Both uploaded PDFs must remain registered in the thread!")
        filenames = {d["filename"] for d in docs}
        self.assertIn("AI Debate Agent Notes.pdf", filenames)
        self.assertIn("MBR_Automation_Project_Documentation.pdf", filenames)

        # 4. Verify PostgreSQL persistence
        db_docs = get_thread_documents_from_db(thread_id, status="ready")
        self.assertEqual(len(db_docs), 2, "PostgreSQL document_metadata must store both records!")

        # 5. Verify isolated directory storage
        path1 = RAG_DATA_DIR / res1["index_path"]
        path2 = RAG_DATA_DIR / res2["index_path"]
        self.assertNotEqual(path1, path2)
        self.assertTrue(path1.is_dir() and (path1 / "index.faiss").is_file())
        self.assertTrue(path2.is_dir() and (path2 / "index.faiss").is_file())

        # 6. Verify retrieval from each document via rag_tool
        config = {"configurable": {"thread_id": thread_id}}
        
        # Query for Document 1 topic
        tool_out_debate = rag_tool.func("debate transcript streaming UI", config)
        self.assertIn("AI Debate Agent Notes.pdf", tool_out_debate)

        # Query for Document 2 topic
        tool_out_mbr = rag_tool.func("MBR automation template slide generation", config)
        self.assertIn("MBR_Automation_Project_Documentation.pdf", tool_out_mbr)

    def test_duplicate_filename_isolation(self):
        """Test uploading PDFs with the same filename.
        Ensures each gets an isolated document_id and index path without overwriting.
        """
        thread_id = f"test-dup-{uuid.uuid4()}"
        self.created_threads.append(thread_id)

        res1 = ingest_rag_document(
            str(self.debate_pdf),
            thread_id,
            filename="Document.pdf",
        )
        res2 = ingest_rag_document(
            str(self.mbr_pdf),
            thread_id,
            filename="Document.pdf",
        )

        self.assertNotEqual(res1["doc_id"], res2["doc_id"])
        self.assertNotEqual(res1["index_path"], res2["index_path"])
        self.assertTrue(is_faiss_index(RAG_DATA_DIR / res1["index_path"]))
        self.assertTrue(is_faiss_index(RAG_DATA_DIR / res2["index_path"]))

    def test_failed_ingestion_does_not_corrupt_existing_ready_document(self):
        """Test that a failed PDF ingestion in a thread does not delete or break existing ready PDFs."""
        thread_id = f"test-fail-clean-{uuid.uuid4()}"
        self.created_threads.append(thread_id)

        # 1. Ingest valid document
        res_valid = ingest_rag_document(
            str(self.debate_pdf),
            thread_id,
            filename="Valid_Document.pdf",
        )
        self.assertEqual(res_valid["status"], "ready")

        # 2. Ingest empty / invalid PDF
        empty_pdf_path = RAG_DATA_DIR / thread_id / "empty.pdf"
        empty_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        empty_pdf_path.write_bytes(b"")

        with self.assertRaises(PdfIngestionError):
            ingest_rag_document(
                str(empty_pdf_path),
                thread_id,
                filename="empty.pdf",
            )

        # 3. Verify ready document is still ready
        ready_docs = get_rag_documents(thread_id, status="ready")
        self.assertEqual(len(ready_docs), 1)
        self.assertEqual(ready_docs[0]["filename"], "Valid_Document.pdf")

        # 4. Verify failed document is recorded as failed
        failed_docs = get_rag_documents(thread_id, status="failed")
        self.assertEqual(len(failed_docs), 1)
        self.assertEqual(failed_docs[0]["filename"], "empty.pdf")

    def test_cross_thread_isolation(self):
        """Test that two separate threads never share documents."""
        thread_a = f"test-iso-a-{uuid.uuid4()}"
        thread_b = f"test-iso-b-{uuid.uuid4()}"
        self.created_threads.extend([thread_a, thread_b])

        ingest_rag_document(str(self.debate_pdf), thread_a, filename="Debate.pdf")
        ingest_rag_document(str(self.mbr_pdf), thread_b, filename="MBR.pdf")

        docs_a = get_rag_documents(thread_a, status="ready")
        docs_b = get_rag_documents(thread_b, status="ready")

        self.assertEqual(len(docs_a), 1)
        self.assertEqual(docs_a[0]["filename"], "Debate.pdf")
        self.assertEqual(len(docs_b), 1)
        self.assertEqual(docs_b[0]["filename"], "MBR.pdf")

        # RAG query in Thread A must not see MBR
        config_a = {"configurable": {"thread_id": thread_a}}
        out_a = rag_tool.func("MBR automation template", config_a)
        self.assertNotIn("MBR.pdf", out_a)


if __name__ == "__main__":
    unittest.main()

