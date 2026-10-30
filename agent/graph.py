"""
LangGraph assembly and compiled chatbot workflow.
"""

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition

from agent.state import ChatState
from agent.nodes import chat_node, tool_node
from database.checkpointing import get_postgres_checkpointer

# 1. Instantiate the StateGraph
graph = StateGraph(ChatState)

# 2. Add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# 3. Add edges and conditional routing
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# 4. Checkpointer & compilation
checkpoint = get_postgres_checkpointer()
chatbot = graph.compile(checkpointer=checkpoint)

__all__ = [
    "graph",
    "checkpoint",
    "chatbot",
]

