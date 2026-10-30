"""
Deterministic tool routing and LangGraph conditional edges.
"""

import re
from typing import Any, Dict, Optional
import uuid
from langgraph.prebuilt import tools_condition


def _get_deterministic_tool_call(
    user_text: str,
    document_context: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return a tool call for intents that should not depend on model tool selection.

    This is a deterministic routing layer in front of the LLM. It keeps the existing
    LangGraph ToolNode workflow, but ensures that obvious tool requests (for
    example current prices, latest news, arithmetic, weather, and PDF questions)
    actually execute the matching tool even when the model does not emit a
    native tool call.
    """
    text = user_text.strip()
    lower = text.lower()
    if not text:
        return None

    def call(name: str, args: dict) -> Dict[str, Any]:
        return {
            "name": name,
            "args": args,
            "id": str(uuid.uuid4()),
            "type": "tool_call",
        }

    # Current stock quotes are different from retail/product prices.
    stock_terms = ("stock price", "share price", "stock quote", "market price of", "ticker price")
    if any(term in lower for term in stock_terms):
        match = re.search(r"\b(?:for|of|ticker|symbol)\s+([A-Z]{1,6})\b", text)
        if not match:
            match = re.search(r"\b([A-Z]{1,6})\b", text)
        if match:
            return call("get_stock_price", {"symbol": match.group(1).upper()})

    # Current weather must be retrieved, not answered from model memory.
    weather_terms = ("weather", "temperature in", "is it raining", "rain in", "forecast in")
    if any(term in lower for term in weather_terms):
        match = re.search(r"\b(?:in|at|for|near)\s+([A-Za-z][A-Za-z .,'-]{1,60}?)(?:[?.!,]|$)", text, re.IGNORECASE)
        location = match.group(1).strip() if match else "Ahmedabad"
        location = re.sub(r"\b(today|now|currently|right now|please|weather)\b", "", location, flags=re.IGNORECASE).strip(" ,.-")
        if location:
            return call("get_current_weather", {"location": location})

    # Web search for recent/current or changing information, including retail prices.
    search_terms = (
        "latest", "current price", "today's price", "today price", "price of",
        "latest price", "latest news", "recent news", "today's news", "current news",
        "search the web", "search online", "official documentation", "recent reviews",
        "release date", "available now", "current specifications", "latest specifications",
        "compare prices", "source urls", "verify this", "what happened today",
    )
    product_or_news = any(term in lower for term in search_terms)
    if product_or_news:
        return call("tavily_search", {"query": text})

    # PDF/document questions should use the indexed document retriever.
    explicit_document_terms = (
        "uploaded pdf", "uploaded document", "in the pdf", "in this pdf",
        "from the pdf", "according to the document", "in the document",
        "uploaded file", "in the uploaded", "csm ppt", "this document",
        "documentation", "this doc", "the document", "the pdf", "this pdf",
        "new pdf", "latest pdf", "attached pdf", "attached document",
        "what is this document about", "what is this doc about",
        "openclaw", "mbr",
    )
    document_follow_up_terms = (
        "explain this", "explain it", "summarize", "summary", "in short",
        "what is this about",
    )
    if any(term in lower for term in explicit_document_terms) or (
        document_context
        and any(term in lower for term in document_follow_up_terms)
    ):
        return call("rag_tool", {"query": text})

    # Stock purchase must go through the existing interrupt/HITL workflow.
    if any(term in lower for term in ("buy shares", "purchase shares", "buy stock", "purchase stock", "buy stocks")):
        symbol_match = re.search(r"\b(?:of|in|for)\s+([A-Z]{1,6})\b", text)
        quantity_match = re.search(r"\b(\d+)\s+(?:shares|stocks)\b", lower)
        if symbol_match and quantity_match:
            return call("purchase_stock", {"symbol": symbol_match.group(1).upper(), "quantity": int(quantity_match.group(1))})

    # Arithmetic expressions / explicit calculation requests use the calculator.
    math_request = any(term in lower for term in (
        "calculate", "calculator", "divided by", "multiply", "multiplied by",
        "percentage of", "percent of", "square root", "what is ", "what's ",
    ))
    has_math_symbols = bool(re.search(r"\d\s*[+*/^%]\s*\d", text))
    if math_request or has_math_symbols:
        # Handle natural-language arithmetic first so trailing instructions
        # such as "verify using the calculator tool" are not passed to eval.
        division = re.search(r"(?i)([\d,.]+)\s+divided by\s+([\d,.]+)", text)
        if division:
            left = division.group(1).replace(",", "")
            right = division.group(2).replace(",", "")
            return call("calculator", {"expression": f"{left} / {right}"})

        multiplication = re.search(r"(?i)([\d,.]+)\s+(?:multiplied by|times)\s+([\d,.]+)", text)
        if multiplication:
            left = multiplication.group(1).replace(",", "")
            right = multiplication.group(2).replace(",", "")
            operator = "*"
            return call("calculator", {"expression": f"{left} {operator} {right}"})

        expression = text
        expression = re.sub(r"(?i)^.*?\b(?:calculate|calculator|what is|what's|compute)\b\s*", "", expression).strip(" ?")
        expression = re.sub(r"(?i)\s+times\s+", " * ", expression)
        # Extract only a math expression, ignoring explanatory text after it.
        match = re.search(r"[0-9][0-9,.]*(?:\s*[+*/%^-]\s*[0-9][0-9,.]*)+", expression)
        if match:
            safe_expression = match.group(0).replace(",", "")
            return call("calculator", {"expression": safe_expression})

    return None


__all__ = [
    "_get_deterministic_tool_call",
    "tools_condition",
]

