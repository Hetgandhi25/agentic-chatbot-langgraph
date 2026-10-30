"""
Input schemas for stock purchase tool.
"""

from pydantic import BaseModel, Field


class PurchaseStockInput(BaseModel):
    symbol: str = Field(
        ...,
        description="The stock ticker symbol to purchase, e.g. 'AAPL' or 'TSLA'",
    )
    quantity: int = Field(
        ...,
        gt=0,
        description="The number of shares to purchase (positive integer)",
    )

