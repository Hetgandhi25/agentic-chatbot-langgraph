"""
Central configuration settings loaded from environment variables and .env file.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def normalize_ollama_url(url: str) -> str:
    """Normalize localhost to 127.0.0.1 on Windows to eliminate IPv6 connect delays."""
    cleaned = (url or "").strip().rstrip("/")
    if "://localhost:" in cleaned:
        return cleaned.replace("://localhost:", "://127.0.0.1:")
    if cleaned.endswith("://localhost"):
        return cleaned.replace("://localhost", "://127.0.0.1")
    return cleaned


@dataclass
class Settings:
    # vLLM LLM Configuration
    vllm_base_url: str = field(
        default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").strip()
    )
    vllm_api_key: str = field(
        default_factory=lambda: os.getenv("VLLM_API_KEY", "EMPTY").strip()
    )
    vllm_model: str = field(
        default_factory=lambda: os.getenv("VLLM_MODEL", "qwen3.8-27b").strip()
    )
    vllm_temperature: float = 0.7
    vllm_timeout: float = 90.0
    vllm_max_retries: int = 1

    # Ollama Embeddings Configuration
    ollama_raw_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    )
    ollama_embedding_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text").strip()
    )

    @property
    def ollama_effective_url(self) -> str:
        return normalize_ollama_url(self.ollama_raw_url)

    # External APIs
    tavily_api_key: str = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", "").strip()
    )
    alpha_vantage_api_key: str = field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    )
    openweather_api_key: str = field(
        default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", "").strip()
    )

    # PostgreSQL Database Configuration
    postgres_host: str = field(
        default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost").strip()
    )
    postgres_port: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432"))
    )
    postgres_db: str = field(
        default_factory=lambda: os.getenv("POSTGRES_DB", "agentic_chatbot").strip()
    )
    postgres_user: str = field(
        default_factory=lambda: os.getenv("POSTGRES_USER", "postgres").strip()
    )
    postgres_password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "")
    )

    # Storage & Resource directories on D:
    base_dir: Path = BASE_DIR
    rag_data_dir: Path = BASE_DIR / "faiss_db"
    temp_dir: Path = BASE_DIR / ".tmp"
    pip_cache_dir: Path = BASE_DIR / ".cache" / "pip"
    ocr_resources_dir: Path = BASE_DIR / "resources" / "ocr"
    tesseract_cmd: str = field(
        default_factory=lambda: os.getenv(
            "TESSERACT_CMD",
            str(BASE_DIR / "resources" / "ocr" / "tesseract" / "tesseract.exe"),
        )
    )
    tessdata_dir: str = field(
        default_factory=lambda: os.getenv(
            "TESSDATA_PREFIX",
            str(BASE_DIR / "resources" / "ocr" / "tesseract" / "tessdata"),
        )
    )


# Ensure process environment uses D: drive temporary files and OCR data
os.environ.setdefault("TEMP", str(BASE_DIR / ".tmp"))
os.environ.setdefault("TMP", str(BASE_DIR / ".tmp"))
os.environ.setdefault(
    "TESSDATA_PREFIX",
    str(BASE_DIR / "resources" / "ocr" / "tesseract" / "tessdata"),
)

settings = Settings()

