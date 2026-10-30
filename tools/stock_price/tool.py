"""
Stock price lookup tool using Alpha Vantage API.
"""

from typing import Any, Dict
import requests
from langchain_core.tools import tool

from app.config import settings
from tools.stock_price.schemas import StockPriceInput


@tool(args_schema=StockPriceInput)
def get_stock_price(symbol: str) -> Dict[str, Any]:
    """
    Fetch the latest stock price for a given symbol using Alpha Vantage.
    """
    api_key = settings.alpha_vantage_api_key

    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY is missing from .env"}

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Stock API request failed: {str(e)}"}

