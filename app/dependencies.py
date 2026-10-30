"""
Dependency injection and service singletons for LLM, Embeddings, and Database.
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from psycopg import Connection
from langgraph.checkpoint.postgres import PostgresSaver

from app.config import settings
from database.postgres import get_postgres_connection
from database.checkpointing import get_postgres_checkpointer

_llm_instance: Optional[ChatOpenAI] = None
_embeddings_instance: Optional[OllamaEmbeddings] = None


def get_llm(new_instance: bool = False) -> ChatOpenAI:
    """Return the configured ChatOpenAI model instance for vLLM."""
    global _llm_instance
    if not new_instance and _llm_instance is not None:
        return _llm_instance

    instance = ChatOpenAI(
        model=settings.vllm_model,
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        temperature=settings.vllm_temperature,
        timeout=settings.vllm_timeout,
        max_retries=settings.vllm_max_retries,
    )
    if not new_instance:
        _llm_instance = instance
    return instance


def get_embeddings(new_instance: bool = False) -> OllamaEmbeddings:
    """Return the configured OllamaEmbeddings instance."""
    global _embeddings_instance
    if not new_instance and _embeddings_instance is not None:
        return _embeddings_instance

    instance = OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_effective_url,
        client_kwargs={"timeout": 60.0},
    )
    if not new_instance:
        _embeddings_instance = instance
    return instance


def get_db_connection() -> Connection:
    """Return shared PostgreSQL database connection."""
    return get_postgres_connection()


def get_checkpointer() -> PostgresSaver:
    """Return shared PostgreSQL LangGraph checkpointer."""
    return get_postgres_checkpointer()

