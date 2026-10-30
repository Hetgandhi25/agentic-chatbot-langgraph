
"""Streamlit interface for the LangGraph chatbot."""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agentic_chatbot_backend import chatbot


# -------------------- Thread Management --------------------

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


def convert_messages(messages):
    history = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        # Handle both text and structured message content
        content = message.content
        if isinstance(content, list):
            content = "\n".join(
                item["text"]
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )

        history.append({"role": role, "content": str(content)})

    return history


# -------------------- Application UI --------------------

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agentic Chatbot with LangGraph")
st.caption("Powered by your vLLM model")


# -------------------- Session Initialization --------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

add_thread(st.session_state["thread_id"])


# -------------------- Sidebar --------------------

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
        messages = load_conversation(thread_id)
        st.session_state["message_history"] = convert_messages(messages)
        st.rerun()


# -------------------- Display Conversation --------------------

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------- Chat Input and Streaming --------------------

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
                message_chunk.content
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages",
                )
                if isinstance(message_chunk, AIMessage)
                and isinstance(message_chunk.content, str)
                and message_chunk.content
            )

            st.session_state["message_history"].append(
                {"role": "assistant", "content": ai_message}
            )

        except Exception as error:
            st.error(f"An error occurred: {error}")
