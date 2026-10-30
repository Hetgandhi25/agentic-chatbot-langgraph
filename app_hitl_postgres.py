from agentic_chatbot_hitl_backend_postgres import (
    chatbot,
    get_all_threads,
    get_rag_documents,
    ingest_rag_document,
    check_ollama_health,
    check_ocr_availability,
    clear_failed_documents,
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.types import Command

import streamlit as st
import uuid
import tempfile
import os
from datetime import datetime


# Generate a unique thread ID for each new conversation
def generate_thread_id():
    existing_threads = set(st.session_state.get("chat_threads", []))
    current_thread_id = st.session_state.get("thread_id")
    if current_thread_id:
        existing_threads.add(current_thread_id)

    thread_id = str(uuid.uuid4())
    while thread_id in existing_threads:
        thread_id = str(uuid.uuid4())
    return thread_id


# Add a new thread ID to the conversation list
def add_thread(thread_id):

    # Keep the sidebar list unique and newest threads easy to reorder.
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # ========================= HITL ADDED =========================
    # Clear any pending human approval request
    st.session_state["pending_hitl"] = None
    # =============================================================

    st.query_params["thread_id"] = st.session_state["thread_id"]


def load_conversation(thread_id):
    """Load a previous conversation from the LangGraph checkpointer safely."""
    try:
        state = chatbot.get_state(
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )
        return state.values.get("messages", [])
    except Exception:
        return []



def get_conversation_title(thread_id):
    """Return a short title based on the conversation's first user message."""
    titles = st.session_state.setdefault("thread_titles_cache", {})
    if thread_id in titles:
        return titles[thread_id]

    try:
        for message in load_conversation(thread_id):
            message_type = getattr(message, "type", None)
            if not (isinstance(message, HumanMessage) or message_type == "human"):
                continue

            content = getattr(message, "content", "")
            if not isinstance(content, str):
                content = str(content)
            content = " ".join(content.split()).strip()
            if not content:
                continue

            max_length = 40
            title = content[:max_length - 1].rstrip() + "…" if len(content) > max_length else content
            titles[thread_id] = title
            return title
    except Exception:
        return None

    return None


def get_thread_activity(thread_id):
    """Return the latest checkpoint time for ordering chats by recent activity."""
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        created_at = getattr(state, "created_at", None)
        if created_at is None:
            return 0.0
        if hasattr(created_at, "timestamp"):
            return float(created_at.timestamp())

        value = str(created_at).strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return 0.0



def sort_threads_by_activity(thread_ids):
    """Return unique conversation IDs with the most recently active first."""
    unique_thread_ids = dict.fromkeys(thread_ids)
    return sorted(
        unique_thread_ids,
        key=lambda thread_id: (get_thread_activity(thread_id), thread_id),
        reverse=True,
    )


def restore_message_history(thread_id):
    """Convert checkpoint messages into the visible chat transcript."""
    temp_messages = []
    for message in load_conversation(thread_id):
        message_type = getattr(message, "type", None)
        if isinstance(message, HumanMessage) or message_type == "human":
            role = "user"
        elif isinstance(message, AIMessage) or message_type == "ai":
            role = "assistant"
        else:
            continue

        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = str(content)
        if content.strip():
            temp_messages.append({"role": role, "content": content})

    return temp_messages


# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.

    Returns:
        The pending Interrupt object, or None.
    """

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # Read the current checkpoint state
        state_snapshot = chatbot.get_state(config)

        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            ()
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # Other LangGraph versions store interrupts inside tasks
        tasks = getattr(
            state_snapshot,
            "tasks",
            ()
        ) or ()

        for task in tasks:

            task_interrupts = getattr(
                task,
                "interrupts",
                ()
            ) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:

        # A newly created thread may not have a checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """

    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.

    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """

    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:

        save_pending_interrupt(
            thread_id,
            pending_interrupt
        )

    else:

        current_pending = st.session_state.get(
            "pending_hitl"
        )

        if (
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    """
    Resume an interrupted LangGraph execution.

    Args:
        decision:
            "yes" approves the stock purchase.
            "no" rejects the stock purchase.
    """

    pending_hitl = st.session_state.get(
        "pending_hitl"
    )

    if not pending_hitl:

        st.warning(
            "There is no pending action to approve or reject."
        )

        return

    # Get the thread that originally triggered the interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # The same thread ID must be used when resuming
    resume_config = {
        "configurable": {
            "thread_id": interrupted_thread_id
        },
        "metadata": {
            "thread_id": interrupted_thread_id
        },
        "run_name": "hitl_resume_trace",
    }

    try:

        # Display the resumed response
        with st.chat_message("assistant"):

            status_holder = {
                "box": st.status(
                    "🔄 Resuming the requested action...",
                    expanded=True
                )
            }

            def resumed_ai_only_stream():

                # Resume the graph with the human decision
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):

                    # Update tool execution status
                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    # Stream only assistant-generated text
                    if isinstance(message_chunk, AIMessage):
                        content = getattr(message_chunk, "content", "")
                        tool_calls = getattr(message_chunk, "tool_calls", None) or []
                        if content and not tool_calls:
                            yield content

            # Display the streamed final answer
            resumed_ai_message = st.write_stream(
                resumed_ai_only_stream()
            )

            # Check whether another interrupt occurred
            next_interrupt = get_pending_interrupt(
                interrupted_thread_id
            )

            if next_interrupt is not None:

                save_pending_interrupt(
                    interrupted_thread_id,
                    next_interrupt
                )

                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )

            else:

                # No more pending approval
                st.session_state["pending_hitl"] = None

                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        # Save the assistant response in Streamlit UI history
        if resumed_ai_message:

            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

        # Rerun so the response appears in normal chat order
        st.rerun()

    except Exception as error:

        st.error(
            f"Could not resume the requested action: {error}"
        )


# ========================= Page configuration =========================

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖"
)

# Display the main application title
st.title("Agentic Chatbot with LangGraph")


# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = list(dict.fromkeys(get_all_threads()))


# Restore the requested conversation, or open the most recently active one.
if "thread_id" not in st.session_state:
    requested_thread_id = st.query_params.get("thread_id")
    if requested_thread_id in st.session_state["chat_threads"]:
        st.session_state["thread_id"] = requested_thread_id
    elif requested_thread_id and get_rag_documents(requested_thread_id):
        st.session_state["thread_id"] = requested_thread_id
    else:
        ordered_threads = sort_threads_by_activity(
            st.session_state["chat_threads"]
        )
        st.session_state["thread_id"] = (
            ordered_threads[0] if ordered_threads else generate_thread_id()
        )


# Load checkpoint messages when the current UI session has no transcript yet.
if "message_history" not in st.session_state:
    if st.session_state["thread_id"] in st.session_state["chat_threads"]:
        st.session_state["message_history"] = restore_message_history(
            st.session_state["thread_id"]
        )
    else:
        st.session_state["message_history"] = []


# ========================= HITL ADDED =========================

# Store the currently pending human approval request
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

# =============================================================

# Show all documents registered for this thread (Ready and Failed)
try:
    ready_rag_documents = get_rag_documents(st.session_state["thread_id"], status="ready")
    failed_rag_documents = get_rag_documents(st.session_state["thread_id"], status="failed")
except Exception as error:
    ready_rag_documents, failed_rag_documents = [], []
    st.error(f"Could not load this chat's PDF status: {error}")

for document in ready_rag_documents:
    details = (
        f"{document['page_count']} pages, "
        f"{document['character_count']:,} extracted characters"
        if document.get("page_count") is not None
        and document.get("character_count") is not None
        else "existing FAISS index"
    )
    st.caption(f"📄 {document['filename']} — Ready ({details})")
    if document.get("diagnostic"):
        st.caption(document["diagnostic"])

for document in failed_rag_documents:
    st.error(
        f"📄 **{document['filename']}** — Failed\n\n"
        f"**Reason:** {document.get('error', 'Extraction failed.')}\n\n"
        "💡 *Please upload a text-searchable PDF (or use a document with selectable text), and try again.*"
    )

st.query_params["thread_id"] = st.session_state["thread_id"]


# ========================= HITL ADDED =========================

# Recover pending approval after page refresh or rerun
sync_pending_interrupt(
    st.session_state["thread_id"]
)

# =============================================================


# ========================= Sidebar threading feature =========================

# Display the sidebar title
st.sidebar.title("My Conversations")


# Create a button for starting a new conversation
if st.sidebar.button("New Chat"):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()



# Display conversations by latest activity; unsent chats remain temporary.
thread_ids = sort_threads_by_activity(st.session_state["chat_threads"])
current_thread_id = st.session_state["thread_id"]

for thread_id in thread_ids:

    conversation_title = get_conversation_title(thread_id)
    if conversation_title is None:
        continue

    # Show a readable conversation title instead of the raw UUID.
    if st.sidebar.button(
        conversation_title,
        key=f"conversation_{thread_id}",
        type="primary" if thread_id == current_thread_id else "secondary",
        use_container_width=True,
    ):

        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = restore_message_history(thread_id)
        st.query_params["thread_id"] = thread_id

        # ========================= HITL ADDED =========================

        # Restore any pending approval for this conversation
        sync_pending_interrupt(thread_id)

        # =============================================================

        # Rerun the application to display the loaded messages
        st.rerun()


# ========================= Main chat interface =========================

# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Render assistant responses as Markdown for proper formatting
        if message["role"] == "assistant":
            st.markdown(message["content"])
        else:
            st.text(message["content"])


# ========================= HITL approval interface =========================

# Get the currently pending approval request
pending_hitl = st.session_state.get(
    "pending_hitl"
)

# Check whether the pending approval belongs to
# the currently selected conversation
current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id")
    == st.session_state["thread_id"]
)


# Display approval controls
if current_thread_has_pending_hitl:

    st.warning(
        "🧑 Human approval required\n\n"
        f"{pending_hitl['prompt']}"
    )

    approve_column, reject_column = st.columns(2)

    # Approve button
    with approve_column:

        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):

            # Send "yes" back to interrupt()
            resume_hitl_execution("yes")

    # Reject button
    with reject_column:

        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True
        ):

            # Send "no" back to interrupt()
            resume_hitl_execution("no")


# ========================= Fixed chat input with PDF upload =========================

# Keep st.chat_input directly in the main body.
# This keeps it fixed at the bottom of the screen.
#
# accept_file=True adds the attachment button inside the chat input.
# file_type=["pdf"] allows PDF files only.
submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"],

    # Disable input while waiting for human approval
    disabled=current_thread_has_pending_hitl
)


# Default user input value
user_input = None


# Process the submitted text and PDF
if submission:

    # Get the text entered by the user
    user_input = submission.text

    # Get the uploaded files
    # This is always a list when accept_file is enabled
    uploaded_files = submission.files

    # Process every selected PDF and register each only after indexing succeeds.
    for uploaded_pdf in uploaded_files:

        # Store the temporary file path
        temporary_file_path = None

        with st.status(
            f"Uploaded {uploaded_pdf.name}",
            expanded=True,
        ) as upload_status:
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temporary_file:
                    temporary_file.write(uploaded_pdf.getvalue())
                    temporary_file_path = temporary_file.name

                def update_upload_status(state, message):
                    upload_status.update(
                        label=f"{state}: {uploaded_pdf.name} — {message}",
                        state="running",
                        expanded=True,
                    )

                indexed_chunks = ingest_rag_document(
                    temporary_file_path,
                    st.session_state["thread_id"],
                    filename=uploaded_pdf.name,
                    progress_callback=update_upload_status,
                )

                method_label = (
                    " (via OCR)"
                    if indexed_chunks.get("extraction_method") == "ocr"
                    else ""
                )
                upload_status.update(
                    label=(
                        f"Ready: {uploaded_pdf.name} — "
                        f"{indexed_chunks['page_count']} pages, "
                        f"{indexed_chunks['character_count']:,} characters{method_label}, "
                        f"{indexed_chunks['chunk_count']} chunks indexed."
                    ),
                    state="complete",
                    expanded=False,
                )
                if indexed_chunks.get("diagnostic"):
                    st.warning(indexed_chunks["diagnostic"])
            except Exception as error:
                upload_status.update(
                    label=f"Failed: {uploaded_pdf.name}",
                    state="error",
                    expanded=True,
                )
                st.error(f"PDF processing failed for {uploaded_pdf.name}: {error}")
            finally:
                if temporary_file_path and os.path.exists(temporary_file_path):
                    for _ in range(5):
                        try:
                            os.unlink(temporary_file_path)
                            break
                        except OSError:
                            import time
                            time.sleep(0.05)



# Run this block after the user submits a text message
if user_input and user_input.strip():

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_input)

    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):

        # Use a mutable holder so the generator can set/modify it
        status_holder = {
            "box": None
        }

        def ai_only_stream():

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                # Lazily create & update the SAME status container
                # when any tool runs
                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True
                        )

                    else:

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    # Tool-call AI messages are internal routing messages, not
                    # user-facing answers. Render only actual text content.
                    content = getattr(message_chunk, "content", "")
                    tool_calls = getattr(message_chunk, "tool_calls", None) or []
                    if content and not tool_calls:
                        yield content

            # ========================= HITL ADDED =========================

            # interrupt() pauses the graph without returning
            # a completed ToolMessage.
            #
            # Inspect the saved checkpoint after streaming ends.
            pending_interrupt = get_pending_interrupt(
                st.session_state["thread_id"]
            )

            if pending_interrupt is not None:

                # Save the interrupt for displaying approval buttons
                save_pending_interrupt(
                    st.session_state["thread_id"],
                    pending_interrupt
                )

                yield (
                    "\n\n⚠️ This stock purchase requires your approval. "
                    "Use the Approve Purchase or Reject Purchase "
                    "button below."
                )

            # =============================================================

        ai_message = st.write_stream(
            ai_only_stream()
        )

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:

            # Check whether execution is waiting for approval
            if get_pending_interrupt(
                st.session_state["thread_id"]
            ) is not None:

                status_holder["box"].update(
                    label="⏸️ Waiting for human approval",
                    state="complete",
                    expanded=False
                )

            else:

                status_holder["box"].update(
                    label="✅ Tool finished",
                    state="complete",
                    expanded=False
                )

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })

    # Persist this thread in the sidebar only after it has a user message.
    add_thread(st.session_state["thread_id"])

    # ========================= HITL ADDED =========================

    # Approval controls are rendered earlier in the script.
    # Rerun so they appear immediately after interrupt().
    if (
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id")
        == st.session_state["thread_id"]
    ):
        st.rerun()

    # =============================================================

    # Force one clean render after the streamed response has been saved.
    # This prevents transient duplicate/shadow chat bubbles left by the
    # streaming UI from remaining in the page after a response completes.
    st.rerun()
