"""
Chat agent state definition for LangGraph.
"""

from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """Conversation state storing the message history with append reducer."""
    messages: Annotated[List[BaseMessage], add_messages]

