"""
Tools package providing modular agent tools and global registry.
"""

from typing import List
from langchain_core.tools import BaseTool

from tools.search import search_tool
from tools.calculator import calculator
from tools.stock_price import get_stock_price
from tools.weather import get_current_weather
from tools.rag import rag_tool
from tools.purchase_stock import purchase_stock
from tools.pdf import PdfIngestionError, check_ocr_availability, extract_pdf_documents

ALL_TOOLS: List[BaseTool] = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool,
    purchase_stock,
]


def get_all_tools() -> List[BaseTool]:
    """Return the list of all registered tools."""
    return list(ALL_TOOLS)


__all__ = [
    "search_tool",
    "calculator",
    "get_stock_price",
    "get_current_weather",
    "rag_tool",
    "purchase_stock",
    "ALL_TOOLS",
    "get_all_tools",
    "PdfIngestionError",
    "check_ocr_availability",
    "extract_pdf_documents",
]

