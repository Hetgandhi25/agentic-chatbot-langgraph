"""
Input schemas for search tool.
"""

from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The web search query string",
    )

