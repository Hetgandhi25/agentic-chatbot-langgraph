"""
RAG retrieval tool scoped to the active conversation thread.
"""

from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from services.document_service import get_rag_documents, get_retriever
from tools.rag.retrieval import retrieve_relevant_documents, format_retrieved_documents


@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """
    Retrieve information only from PDFs uploaded in the current conversation.

    Use this tool for questions about an uploaded PDF/document. Each chat has
    its own FAISS index, so documents from other conversations are not used.

    Args:
        query: The question or search query used to retrieve PDF content.
        config: LangGraph runtime configuration used to identify the chat.
    """
    thread_id = (config or {}).get("configurable", {}).get("thread_id")

    if not thread_id:
        return (
            "I can't search a PDF because this conversation has no valid "
            "thread ID. Please start a new chat and upload the PDF again."
        )

    ready_docs = get_rag_documents(thread_id, status="ready")
    failed_docs = get_rag_documents(thread_id, status="failed")

    lower_query = query.lower()

    # Detect if query refers to the newest / latest / recent PDF
    new_doc_keywords = (
        "new pdf", "new document", "new doc", "new file",
        "latest pdf", "latest document", "latest doc", "latest file",
        "recent pdf", "recent document", "last pdf", "last document",
        "3rd pdf", "third pdf", "2nd pdf", "second pdf",
    )
    is_asking_for_new_doc = any(phrase in lower_query for phrase in new_doc_keywords)

    # 1. Check if user is asking for the new/latest document while the latest upload failed
    if failed_docs and is_asking_for_new_doc:
        latest_failed = failed_docs[-1]
        filename = latest_failed.get("filename", "the uploaded document")
        failure_reason = latest_failed.get("error", "Text extraction failed.")
        ready_names = [d.get("filename", "Document") for d in ready_docs]
        ready_context = f" Earlier indexed document(s) available in this chat: {', '.join(ready_names)}." if ready_names else ""
        return (
            f"The newly uploaded PDF ('{filename}') could not be processed: {failure_reason} "
            f"Because it could not be indexed, I cannot answer questions from it yet.{ready_context} "
            "Please upload a valid, text-searchable PDF."
        )

    # 2. Check if user mentions a specific failed document by name
    for f_doc in failed_docs:
        f_name = f_doc.get("filename", "")
        f_stem = Path(f_name).stem.lower().replace("_", " ").replace("-", " ")
        f_keywords = [w for w in f_stem.split() if len(w) > 2 and w not in ("technical", "documentation", "project", "automation", "pdf")]
        if f_keywords and any(kw in lower_query for kw in f_keywords):
            f_reason = f_doc.get("error", "Processing failed.")
            return (
                f"The document '{f_name}' could not be processed: {f_reason} "
                "Because text could not be extracted or indexed, I cannot answer questions from its content. "
                "Please upload a clean copy of this document."
            )

    # 3. If we have ready documents, perform retrieval
    if ready_docs:
        target_doc_id = None
        target_doc_name = None

        # If user explicitly asked for the "new pdf" / "latest pdf", target the most recent ready document
        if is_asking_for_new_doc:
            target_doc = ready_docs[-1]
            target_doc_id = target_doc.get("doc_id")
            target_doc_name = target_doc.get("filename")

        # Check if query matches a specific ready document name
        if not target_doc_id:
            for doc in ready_docs:
                fname = doc.get("filename", "")
                stem = Path(fname).stem.lower().replace("_", " ").replace("-", " ")
                keywords = [w for w in stem.split() if len(w) > 2 and w not in ("technical", "documentation", "project", "automation", "pdf")]
                if keywords and any(kw in lower_query for kw in keywords):
                    target_doc_id = doc.get("doc_id")
                    target_doc_name = doc.get("filename")
                    break

        retriever = get_retriever(thread_id, target_doc_id=target_doc_id)
        if retriever is None:
            return (
                "A PDF is registered for this chat, but its index could not be "
                "loaded. Please upload the PDF again."
            )

        index = retriever.vectorstore
        k = 4 if target_doc_id else min(12, max(4, 3 * len(ready_docs)))
        documents = retrieve_relevant_documents(index, query, k=k, score_threshold=0.06)

        # If strict threshold returned nothing, retry with a more permissive threshold to avoid empty answers
        if not documents:
            documents = retrieve_relevant_documents(index, query, k=k, score_threshold=0.01)

        if not documents:
            return (
                "No relevant information was found in the indexed PDF(s) "
                "uploaded to this chat for your query. The document content "
                "does not appear to support an answer to this question."
            )

        formatted = format_retrieved_documents(documents)
        if len(ready_docs) > 1 and not target_doc_id:
            active_list = ", ".join(d.get("filename", "Doc") for d in ready_docs)
            return f"[Active Documents in this conversation: {active_list}]\n\n{formatted}"
        return formatted

    # 4. If no ready docs, but failed docs exist
    if failed_docs:
        latest_failed = failed_docs[-1]
        filename = latest_failed.get("filename", "the uploaded document")
        failure_reason = latest_failed.get("error", "Text extraction produced 0 characters.")
        return (
            f"A PDF ('{filename}') was uploaded to this conversation, but processing failed: {failure_reason} "
            "Because text could not be extracted or indexed, I cannot answer questions from its content yet. "
            "Please upload an unlocked, text-searchable PDF (or use a document with selectable text), and try again."
        )

    # 5. No document (neither ready nor failed) has been uploaded in this thread
    return (
        "I can't answer this from a PDF yet because no PDF has been "
        "uploaded and indexed in this chat. Please attach the relevant "
        "PDF using the + button in the chat input, wait for the upload "
        "confirmation, and ask your question again. You can still ask "
        "general questions, request web searches, or ask about weather "
        "without uploading a PDF."
    )

