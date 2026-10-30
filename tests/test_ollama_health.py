import os
import unittest
import requests
from unittest.mock import patch, MagicMock

# Load environment
from dotenv import load_dotenv
load_dotenv()

class TestOllamaDiagnostics(unittest.TestCase):
    def test_health_check_live(self):
        import agentic_chatbot_hitl_backend_postgres as backend
        result = backend.check_ollama_health()
        self.assertEqual(result["status"], "healthy")
        self.assertIn("nomic-embed-text", result["model"])

    def test_health_check_unreachable_endpoint(self):
        import agentic_chatbot_hitl_backend_postgres as backend
        with patch.object(backend, "get_effective_ollama_config", return_value=("http://127.0.0.1:9999", "nomic-embed-text")):
            with self.assertRaises(backend.PdfIngestionError) as ctx:
                backend.check_ollama_health(timeout=1.0)
            self.assertIn("Failed to connect to Ollama", str(ctx.exception))

    def test_health_check_missing_model(self):
        import agentic_chatbot_hitl_backend_postgres as backend
        with patch.object(backend, "get_effective_ollama_config", return_value=("http://127.0.0.1:11434", "non-existent-model-xyz")):
            with self.assertRaises(backend.PdfIngestionError) as ctx:
                backend.check_ollama_health(timeout=2.0)
            self.assertIn("not available on Ollama server", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()

