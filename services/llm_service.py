"""
LLM model service and runtime binding.
"""

from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool

from app.dependencies import get_llm


def get_llm_service(new_instance: bool = False) -> ChatOpenAI:
    """Return the primary LLM model instance."""
    return get_llm(new_instance=new_instance)


def get_llm_with_tools(tools: List[BaseTool], llm: Optional[ChatOpenAI] = None):
    """Bind the given tools to the LLM model instance."""
    model = llm or get_llm_service()
    return model.bind_tools(tools)

