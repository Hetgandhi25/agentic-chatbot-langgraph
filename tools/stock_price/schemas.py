"""
Input schemas for stock price lookup tool.
"""

from pydantic import BaseModel, Field


class StockPriceInput(BaseModel):
    symbol: str = Field(
        ...,
        description="The stock ticker symbol to look up, e.g. 'AAPL' or 'TSLA'",
    )

