"""
LangGraph execution nodes: chat_node and tool_node.
"""

from typing import Any, Dict
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from agent.state import ChatState
from agent.prompts import get_system_message
from agent.routing import _get_deterministic_tool_call
from agent.tool_registry import list_registered_tools
from services.llm_service import get_llm_with_tools
from services.document_service import has_rag_index
from storage.document_metadata import read_document_manifest

tools = list_registered_tools()
llm_with_tools = get_llm_with_tools(tools)
tool_node = ToolNode(tools)


def chat_node(state: ChatState, config: RunnableConfig) -> Dict[str, Any]:
    """Use deterministic routing for clear tool intents, then let the LLM answer."""

    # Only inspect a fresh user message. After a tool runs, the latest message
    # is a ToolMessage and the LLM must interpret the tool result instead.
    latest_message = state["messages"][-1] if state.get("messages") else None
    if isinstance(latest_message, HumanMessage):
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        previous_rag_call = any(
            getattr(message, "name", None) == "rag_tool"
            or any(
                tool_call.get("name") == "rag_tool"
                for tool_call in (
                    getattr(message, "tool_calls", None) or []
                )
                if isinstance(tool_call, dict)
            )
            for message in state.get("messages", [])[:-1]
        )
        has_thread_docs = bool(thread_id) and (
            has_rag_index(thread_id) or bool(read_document_manifest(thread_id))
        )
        document_context = previous_rag_call or has_thread_docs
        forced_tool_call = _get_deterministic_tool_call(
            str(latest_message.content),
            document_context=document_context,
        )
        if forced_tool_call is not None:
            return {"messages": [AIMessage(content="", tool_calls=[forced_tool_call])]}

    system_message = get_system_message()
    messages = [
        system_message,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


__all__ = [
    "chat_node",
    "tool_node",
    "llm_with_tools",
    "tools",
]

