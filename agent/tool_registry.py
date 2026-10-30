"""
Central registry of tools available to the LangGraph agent.
"""

from typing import Dict, List, Optional
from langchain_core.tools import BaseTool

from tools import (
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool,
    purchase_stock,
    get_all_tools,
)

# Registry dictionary mapping tool name to tool instance
TOOL_REGISTRY: Dict[str, BaseTool] = {
    t.name: t for t in get_all_tools()
}


def get_tool(name: str) -> Optional[BaseTool]:
    """Retrieve a tool by its registered name."""
    return TOOL_REGISTRY.get(name)


def list_registered_tools() -> List[BaseTool]:
    """Return all registered agent tools."""
    return get_all_tools()


__all__ = [
    "TOOL_REGISTRY",
    "get_tool",
    "list_registered_tools",
    "search_tool",
    "calculator",
    "get_stock_price",
    "get_current_weather",
    "rag_tool",
    "purchase_stock",
]

