"""
Unit tests for RAG tool.
"""

import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from tools.rag import rag_tool, format_retrieved_documents


class TestRagTool(unittest.TestCase):
    def test_missing_thread_id(self):
        res = rag_tool.invoke({"query": "What is in the PDF?"}, config={"configurable": {}})
        self.assertIn("no valid thread ID", res)

    @patch("tools.rag.tool.get_rag_documents")
    def test_empty_thread_prompts_upload(self, mock_get_docs):
        mock_get_docs.return_value = []
        res = rag_tool.invoke(
            {"query": "What is in the PDF?"},
            config={"configurable": {"thread_id": "fresh-thread-123"}},
        )
        self.assertIn("no PDF has been uploaded and indexed in this chat", res)

    @patch("tools.rag.tool.get_rag_documents")
    def test_failed_document_informs_user(self, mock_get_docs):
        def side_effect(thread_id, status=None):
            if status == "ready":
                return []
            if status == "failed":
                return [{
                    "filename": "test.pdf",
                    "error": "The PDF has 5 pages and 0 extracted characters.",
                    "status": "failed",
                }]
            return []

        mock_get_docs.side_effect = side_effect
        res = rag_tool.invoke(
            {"query": "explain this pdf"},
            config={"configurable": {"thread_id": "failed-thread-123"}},
        )
        self.assertIn("processing failed: The PDF has 5 pages and 0 extracted characters", res)

    def test_format_retrieved_documents(self):
        docs = [
            Document(page_content="Text from page 1", metadata={"source": "report.pdf", "page": 1}),
            Document(page_content="Text from page 2", metadata={"source": "report.pdf", "page": 2}),
        ]
        formatted = format_retrieved_documents(docs)
        self.assertIn("Document 1", formatted)
        self.assertIn("Source: report.pdf", formatted)
        self.assertIn("Page: 1", formatted)
        self.assertIn("Text from page 1", formatted)
        self.assertIn("Document 2", formatted)


if __name__ == "__main__":
    unittest.main()

