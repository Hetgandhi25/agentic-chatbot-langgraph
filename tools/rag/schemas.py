"""
Input schemas for RAG retrieval tool.
"""

from pydantic import BaseModel, Field


class RagInput(BaseModel):
    query: str = Field(
        ...,
        description="The question or search query used to retrieve PDF content from the active conversation",
    )

