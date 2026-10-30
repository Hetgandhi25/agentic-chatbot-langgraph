
"""Streamlit chat interface with persistent conversation history."""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agentic_chatbot_db_backend import chatbot, get_all_threads


def generate_thread_id() -> str:
    """Generate a unique ID for a conversation."""
    return str(uuid.uuid4())


def add_thread(thread_id: str) -> None:
    """Add a thread to the sidebar list if it is not already present."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat() -> None:
    """Start a new conversation."""
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id: str) -> list:
    """Load saved messages from the SQLite checkpoint."""
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


def convert_messages(messages: list) -> list[dict]:
    """Convert LangChain messages to Streamlit chat history."""
    history = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        content = message.content

        if isinstance(content, list):
            content = "\n".join(
                item["text"]
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
            )

        history.append({"role": role, "content": str(content)})

    return history


st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("Agentic Chatbot with LangGraph")
st.caption("Powered by vLLM and persistent SQLite memory")


# Initialize Streamlit session state.
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

add_thread(st.session_state["thread_id"])


# Sidebar: create and select conversations.
st.sidebar.title("My Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

for thread_id in reversed(st.session_state["chat_threads"]):
    if st.sidebar.button(
        f"Chat {thread_id[:8]}",
        key=f"thread_{thread_id}",
        use_container_width=True,
    ):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = convert_messages(
            load_conversation(thread_id)
        )
        st.rerun()


# Display the selected conversation.
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Stream the assistant's response.
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    with st.chat_message("assistant"):
        try:
            ai_message = st.write_stream(
                chunk.content
                for chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages",
                )
                if isinstance(chunk, AIMessage)
                and isinstance(chunk.content, str)
                and chunk.content
            )

            st.session_state["message_history"].append(
                {"role": "assistant", "content": ai_message}
            )

        except Exception:
            st.error(
                "The response failed. Check the VS Code terminal "
                "for the underlying error."
            )
            raise
