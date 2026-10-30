"""
Stock purchase tool with Human-In-The-Loop (HITL) approval via LangGraph interrupt.
"""

from typing import Any, Dict
from langchain_core.tools import tool
from langgraph.types import interrupt

from tools.purchase_stock.schemas import PurchaseStockInput


@tool(args_schema=PurchaseStockInput)
def purchase_stock(symbol: str, quantity: int) -> Dict[str, Any]:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, interrupt and wait for a
    human decision ("yes" / anything else).
    """
    decision = interrupt(
        f"Approve buying {quantity} shares of {symbol}? (yes/no)"
    )

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }

