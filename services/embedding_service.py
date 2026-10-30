"""
Ollama embedding service and health validation utilities.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import requests

from app.config import settings, normalize_ollama_url
from tools.pdf.schemas import PdfIngestionError


class EmbeddingServiceError(PdfIngestionError):
    """Raised when the embedding service is unreachable or unconfigured."""


def get_effective_ollama_config() -> Tuple[str, str]:
    """Return the raw configured base URL and embedding model name."""
    return settings.ollama_raw_url, settings.ollama_embedding_model


def check_ollama_health(
    timeout: float = 5.0,
    config_getter: Optional[Callable[[], Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Verify that the Ollama embedding service is running and the configured model is installed.

    Raises:
        PdfIngestionError: If the server cannot be reached, times out, or the model is missing.
    """
    getter = config_getter or get_effective_ollama_config
    raw_url, model = getter()
    target_url = normalize_ollama_url(raw_url)

    # 1. Reachability check
    try:
        response = requests.get(f"{target_url.rstrip('/')}/api/tags", timeout=timeout)
    except (requests.ConnectionError, requests.Timeout) as error:
        is_local = "127.0.0.1" in target_url or "localhost" in target_url
        if is_local:
            help_msg = (
                f"Failed to connect to Ollama at {raw_url}. "
                "Please verify that Ollama is running on your machine. "
                "You can start it by running 'ollama serve' in PowerShell or launching Ollama from the Start menu."
            )
        else:
            help_msg = (
                f"Failed to connect to remote Ollama server at {raw_url}. "
                "Please verify network connectivity and ensure Ollama is accessible from this machine."
            )
        raise PdfIngestionError(help_msg) from error
    except Exception as error:
        raise PdfIngestionError(
            f"Unexpected error connecting to Ollama at {raw_url}: {error}"
        ) from error

    if response.status_code != 200:
        raise PdfIngestionError(
            f"Ollama server at {raw_url} returned HTTP {response.status_code}: {response.text}"
        )

    try:
        tags_data = response.json()
    except Exception as error:
        raise PdfIngestionError(
            f"Ollama server at {raw_url} returned invalid JSON: {error}"
        ) from error

    raw_models = tags_data.get("models", [])
    model_names: List[str] = [m.get("name", "") for m in raw_models if isinstance(m, dict)]

    # 2. Model availability check (matches 'nomic-embed-text' with 'nomic-embed-text:latest' etc.)
    base_target = model.split(":")[0]
    matched = any(
        name == model
        or name == f"{model}:latest"
        or name.split(":")[0] == base_target
        for name in model_names
    )

    if not matched:
        available = ", ".join(model_names) if model_names else "none"
        raise PdfIngestionError(
            f"The embedding model '{model}' is not available on Ollama server ({raw_url}). "
            f"Currently installed models: [{available}]. "
            f"Please install it by running: ollama pull {model}"
        )

    # 3. Quick embedding probe (ensures model is loaded and responding)
    try:
        probe_res = requests.post(
            f"{target_url.rstrip('/')}/api/embed",
            json={"model": model, "input": "health check probe"},
            timeout=timeout,
        )
        if probe_res.status_code != 200:
            probe_res = requests.post(
                f"{target_url.rstrip('/')}/api/embeddings",
                json={"model": model, "prompt": "health check probe"},
                timeout=timeout,
            )
        if probe_res.status_code != 200:
            raise PdfIngestionError(
                f"Ollama embedding model '{model}' returned HTTP {probe_res.status_code}: {probe_res.text}"
            )
    except (requests.ConnectionError, requests.Timeout) as error:
        raise PdfIngestionError(
            f"Ollama model probe for '{model}' timed out or failed to respond: {error}"
        ) from error

    return {
        "status": "healthy",
        "base_url": raw_url,
        "effective_url": target_url,
        "model": model,
        "installed_models": model_names,
    }
