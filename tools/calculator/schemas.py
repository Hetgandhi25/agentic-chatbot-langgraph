"""
Input schemas for calculator tool.
"""

from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    expression: str = Field(
        ...,
        description="A valid mathematical expression to evaluate, e.g. '2 + 2' or 'math.sqrt(16)'",
    )

