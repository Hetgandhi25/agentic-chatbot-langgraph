
"""LangGraph chatbot backend with persistent SQLite checkpoints."""

import os
import sqlite3
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# Use your existing OpenAI-compatible vLLM endpoint.
llm = ChatOpenAI(
    model=os.environ["VLLM_MODEL"],
    base_url=os.environ["VLLM_BASE_URL"],
    api_key=os.environ["VLLM_API_KEY"],
    temperature=0,
)


class ChatState(TypedDict):
    """State containing the conversation messages."""

    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> dict:
    """Generate an assistant response using the configured vLLM model."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Keep the database in the project directory regardless of the launch path.
DATABASE_PATH = Path(__file__).resolve().parent / "chatbot.db"

# check_same_thread=False allows SQLite access across Streamlit execution threads.
conn = sqlite3.connect(
    str(DATABASE_PATH),
    check_same_thread=False,
    timeout=30,
)

checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Create the persistent LangGraph chatbot.
chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads() -> list[str]:
    """Return unique conversation thread IDs stored in SQLite."""
    return sorted(
        {
            item.config["configurable"]["thread_id"]
            for item in checkpoint.list(None)
            if item.config.get("configurable", {}).get("thread_id")
        }
    )
