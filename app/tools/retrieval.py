
"""Document retrieval tools for the Agentic RAG system."""

from langchain_core.tools import tool

from app import config
from app.llm import get_llm
from app.tools.shared import (
    get_store,
    resolve_document,
    not_found_message,
    format_results,
    to_sources,
)


@tool(response_format="content_and_artifact")
def search_documents(query: str, document: str = ""):
    """
    Search uploaded documents for relevant passages.

    Use this tool before answering factual questions about uploaded files.
    Provide a natural-language search query.
    Optionally specify a document filename.
    """
    query = (query or "").strip()
    document = (document or "").strip()

    if not query:
        return "Please provide a search query.", []

    store = get_store()

    source = None

    if document:
        source = resolve_document(document)

        if not source:
            return not_found_message(document), []

    results = store.search(
        query=query,
        k=config.TOP_K,
        source=source,
    )

    if not results:
        return (
            "No relevant passages were found in the uploaded documents.",
            [],
        )

    formatted = format_results(results)
    sources = to_sources(results)

    return formatted, sources


@tool
def list_documents() -> str:
    """List all uploaded documents in the knowledge base."""
    store = get_store()
    documents = store.list_documents()

    if not documents:
        return "No documents have been uploaded yet."

    lines = ["Uploaded documents:"]

    for document in documents:
        if isinstance(document, dict):
            name = document.get("source", document.get("name", "Unknown"))
            count = document.get("chunks", document.get("count"))

            if count is not None:
                lines.append(f"- {name} ({count} chunks)")
            else:
                lines.append(f"- {name}")
        else:
            lines.append(f"- {document}")

    return "\n".join(lines)


@tool
def rewrite_query(question: str) -> str:
    """
    Rewrite a question into alternative search queries.

    Use this when the first document search does not find useful passages.
    This tool only suggests queries; search_documents must still be called.
    """
    question = (question or "").strip()

    if not question:
        return "Please provide a question to rewrite."

    prompt = f"""
Rewrite the following question into three concise alternative
document-search queries.

Keep the original meaning.
Use important keywords and relevant synonyms.
Return only the three queries, one per line.
Do not answer the question.

Question:
{question}
"""

    try:
        response = get_llm(fast=True).invoke(prompt)
        return str(response.content).strip()

    except Exception as exc:
        return f"Could not rewrite the search query: {exc}"
