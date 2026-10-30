import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, ToolMessage
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vector = [0.0] * 4096
        for token in text.lower().split():
            index = int.from_bytes(
                hashlib.sha256(token.encode("utf-8")).digest()[:2],
                byteorder="big",
            ) % len(vector)
            vector[index] += 1.0
        magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / magnitude for value in vector]


def write_pdf(path: Path, text: str | None = None, password: str | None = None):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    if text is not None:
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({
                NameObject("/F1"): writer._add_object(font),
            }),
        })
        content = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(
            f"BT /F1 12 Tf 72 720 Td ({content}) Tj ET".encode("latin-1")
        )
        page[NameObject("/Contents")] = writer._add_object(stream)

    if password:
        writer.encrypt(password)
    with path.open("wb") as pdf_file:
        writer.write(pdf_file)


def write_image_only_pdf(path: Path):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    image = DecodedStreamObject()
    image.set_data(bytes((255, 0, 0)))
    image.update({
        NameObject("/Type"): NameObject("/XObject"),
        NameObject("/Subtype"): NameObject("/Image"),
        NameObject("/Width"): NumberObject(1),
        NameObject("/Height"): NumberObject(1),
        NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
        NameObject("/BitsPerComponent"): NumberObject(8),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/XObject"): DictionaryObject({
            NameObject("/Im0"): writer._add_object(image),
        }),
    })
    content = DecodedStreamObject()
    content.set_data(b"q 100 0 0 100 72 650 cm /Im0 Do Q")
    page[NameObject("/Contents")] = writer._add_object(content)
    with path.open("wb") as pdf_file:
        writer.write(pdf_file)


class RagPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import langgraph.checkpoint.postgres as postgres_module
        from langgraph.checkpoint.memory import InMemorySaver
        import psycopg

        class FakePostgresSaver(InMemorySaver):
            def __init__(self, connection):
                super().__init__()
                self.connection = connection

            def setup(self):
                pass

            def list(self, config):
                return []

        with patch.object(psycopg, "connect", return_value=MagicMock()):
            with patch.object(
                postgres_module,
                "PostgresSaver",
                FakePostgresSaver,
            ):
                import agentic_chatbot_hitl_backend_postgres as backend

        cls.backend = backend

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.data_path = Path(self.temporary_directory.name) / "faiss_db"
        self.rag_path_patch = patch.object(
            self.backend,
            "RAG_DATA_DIR",
            self.data_path,
        )
        self.rag_path_patch.start()
        self.addCleanup(self.rag_path_patch.stop)
        self.embedding_patch = patch.object(
            self.backend,
            "embeddings",
            FakeEmbeddings(),
        )
        self.embedding_patch.start()
        self.addCleanup(self.embedding_patch.stop)

    def test_text_pdf_is_indexed_persisted_and_retrievable(self):
        thread_id = "thread-text"
        pdf_path = Path(self.temporary_directory.name) / "guide.pdf"
        write_pdf(
            pdf_path,
            "The project codebook establishes a weekly safety review.",
        )

        result = self.backend.ingest_rag_document(
            str(pdf_path),
            thread_id,
            filename="guide.pdf",
        )

        self.assertEqual(result["page_count"], 1)
        self.assertGreater(result["character_count"], 0)
        self.assertGreater(result["chunk_count"], 0)
        self.assertTrue(self.backend.has_rag_index(thread_id))
        self.assertEqual(
            [entry["filename"] for entry in self.backend.get_rag_documents(thread_id)],
            ["guide.pdf"],
        )
        response = self.backend.rag_tool.func(
            "weekly safety review",
            {"configurable": {"thread_id": thread_id}},
        )
        self.assertIn("weekly safety review", response.lower())
        self.assertIn("guide.pdf", response)
        irrelevant_response = self.backend.rag_tool.func(
            "quantum banana constellation",
            {"configurable": {"thread_id": thread_id}},
        )
        self.assertIn("no relevant information", irrelevant_response.lower())

    def test_multiple_documents_are_retrievable_only_in_their_thread(self):
        first_thread = "thread-first"
        other_thread = "thread-other"
        first_pdf = Path(self.temporary_directory.name) / "first.pdf"
        second_pdf = Path(self.temporary_directory.name) / "second.pdf"
        write_pdf(first_pdf, "The cobalt manual describes a weekly safety review.")
        write_pdf(second_pdf, "The amber guide describes a monthly garden plan.")

        self.backend.ingest_rag_document(
            str(first_pdf), first_thread, filename="first.pdf"
        )
        self.backend.ingest_rag_document(
            str(second_pdf), first_thread, filename="second.pdf"
        )

        first_response = self.backend.rag_tool.func(
            "What does the cobalt manual say about safety review?",
            {"configurable": {"thread_id": first_thread}},
        )
        other_response = self.backend.rag_tool.func(
            "What does the cobalt manual say about safety review?",
            {"configurable": {"thread_id": other_thread}},
        )
        self.assertIn("weekly safety review", first_response.lower())
        self.assertIn("no pdf has been uploaded", other_response.lower())
        self.assertEqual(
            len(self.backend.get_rag_documents(first_thread)),
            2,
        )

    def test_invalid_empty_encrypted_and_blank_pdfs_are_not_registered(self):
        thread_id = "thread-invalid"
        invalid_pdf = Path(self.temporary_directory.name) / "invalid.pdf"
        empty_pdf = Path(self.temporary_directory.name) / "empty.pdf"
        encrypted_pdf = Path(self.temporary_directory.name) / "encrypted.pdf"
        blank_pdf = Path(self.temporary_directory.name) / "blank.pdf"
        malformed_pdf = Path(self.temporary_directory.name) / "malformed.pdf"
        invalid_pdf.write_bytes(b"not a PDF")
        empty_pdf.write_bytes(b"")
        malformed_pdf.write_bytes(b"%PDF-1.7\n")
        write_pdf(encrypted_pdf, "secret text", password="test_dummy_password")
        write_pdf(blank_pdf)

        for pdf_path, error_text in (
            (invalid_pdf, "not a valid PDF"),
            (empty_pdf, "empty"),
            (malformed_pdf, "Could not open the PDF"),
            (encrypted_pdf, "password-protected"),
            (blank_pdf, "0 extracted text characters"),
        ):
            with self.subTest(pdf=pdf_path.name):
                with self.assertRaisesRegex(ValueError, error_text):
                    self.backend.ingest_rag_document(
                        str(pdf_path),
                        thread_id,
                        filename=pdf_path.name,
                    )
                self.assertFalse(self.backend.has_rag_index(thread_id))

    def test_index_or_manifest_failure_does_not_register_document(self):
        thread_id = "thread-index-failure"
        pdf_path = Path(self.temporary_directory.name) / "guide.pdf"
        write_pdf(pdf_path, "A valid document that should not be registered.")

        with patch.object(
            self.backend.FAISS,
            "from_documents",
            side_effect=RuntimeError("embedding service unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "embedding service unavailable"):
                self.backend.ingest_rag_document(
                    str(pdf_path), thread_id, filename="guide.pdf"
                )
        self.assertFalse(self.backend.has_rag_index(thread_id))

        with patch.object(
            self.backend,
            "_write_document_manifest",
            side_effect=OSError("manifest write failed"),
        ):
            with self.assertRaisesRegex(OSError, "manifest write failed"):
                self.backend.ingest_rag_document(
                    str(pdf_path), thread_id, filename="guide.pdf"
                )
        self.assertFalse(self.backend.has_rag_index(thread_id))

    def test_image_only_pdf_reports_ocr_limitation(self):
        scanned_pdf = Path(self.temporary_directory.name) / "scan.pdf"
        write_image_only_pdf(scanned_pdf)
        with self.assertRaisesRegex(
            ValueError,
            r"OCR is unavailable|Both native text extraction and OCR",
        ) as error:
            self.backend.ingest_rag_document(
                str(scanned_pdf),
                "thread-scanned",
                filename="scan.pdf",
            )
        self.assertIn("1 page", str(error.exception))
        self.assertIn("0 extracted text characters", str(error.exception))
        self.assertFalse(self.backend.has_rag_index("thread-scanned"))

    def test_document_requests_are_routed_to_rag(self):
        for query in (
            "Explain the documentation.",
            "What is this document about?",
        ):
            with self.subTest(query=query):
                call = self.backend._get_deterministic_tool_call(query)
                self.assertIsNotNone(call)
                self.assertEqual(call["name"], "rag_tool")

        for query in ("Explain in short.", "Explain this in short."):
            with self.subTest(query=query):
                self.assertIsNone(
                    self.backend._get_deterministic_tool_call(query)
                )
                call = self.backend._get_deterministic_tool_call(
                    query,
                    document_context=True,
                )
                self.assertEqual(call["name"], "rag_tool")

    def test_chat_node_uses_current_document_or_recent_rag_context(self):
        thread_id = "thread-context"
        with patch.object(self.backend, "has_rag_index", return_value=True):
            indexed_call = self.backend.chat_node(
                {"messages": [HumanMessage(content="Explain in short.")]},
                {"configurable": {"thread_id": thread_id}},
            )
        self.assertEqual(
            indexed_call["messages"][0].tool_calls[0]["name"],
            "rag_tool",
        )

        with patch.object(self.backend, "has_rag_index", return_value=False):
            follow_up_call = self.backend.chat_node(
                {
                    "messages": [
                        ToolMessage(
                            content="No document is indexed.",
                            tool_call_id="rag-call",
                            name="rag_tool",
                        ),
                        HumanMessage(
                            content="Explain this in short."
                        ),
                    ],
                },
                {"configurable": {"thread_id": thread_id}},
            )
        self.assertEqual(
            follow_up_call["messages"][0].tool_calls[0]["name"],
            "rag_tool",
        )


if __name__ == "__main__":
    unittest.main()
