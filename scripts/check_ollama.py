"""
Diagnostic script to check Ollama daemon health, model availability, and embedding generation.
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from app.config import settings
from services.embedding_service import get_effective_ollama_config, check_ollama_health
from app.dependencies import get_embeddings


def main():
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("========================================")
    print("OLLAMA SERVICE DIAGNOSTICS")
    print("========================================")

    raw_url, model = get_effective_ollama_config()
    print(f"Configured Raw URL: {raw_url}")
    print(f"Effective URL:     {settings.ollama_effective_url}")
    print(f"Embedding Model:   {model}")

    try:
        health = check_ollama_health(timeout=5.0)
        print("\n[PASS] Ollama service is reachable and healthy.")
        print(f"Installed models: {', '.join(health.get('installed_models', []))}")
    except Exception as e:
        print(f"\n[FAIL] Ollama health check failed: {e}")
        sys.exit(1)

    print("\nTesting embedding generation with sample text...")
    try:
        embeddings = get_embeddings()
        sample_text = "Diagnostic probe for Ollama embedding service."
        vector = embeddings.embed_query(sample_text)
        print(f"[PASS] Successfully generated embedding vector (dimensions: {len(vector)}).")
    except Exception as e:
        print(f"[FAIL] Embedding generation failed: {e}")
        sys.exit(1)

    print("\nAll Ollama diagnostic checks passed!")


if __name__ == "__main__":
    main()

