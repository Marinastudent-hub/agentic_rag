"""Retrieval tools: list_documents, search_documents, rewrite_query."""
import re

from langchain_core.tools import tool

from app import config
from app.llm import get_llm
from app.tools.shared import (
    get_store, resolve_document, not_found_message, format_results, to_sources,
)


@tool
def list_documents() -> str:
    """List every document in the knowledge base with its page and section counts.
    Use this when the user asks what documents are available, or when you are
    unsure of the exact name of a document."""
    docs = get_store().list_documents()
    if not docs:
        return "The knowledge base is empty. No documents have been uploaded."
    lines = [f"- {d['source']} ({d['pages']} pages, {d['chunks']} sections)"
             for d in docs]
    return f"{len(docs)} document(s) in the knowledge base:\n" + "\n".join(lines)


@tool(response_format="content_and_artifact")
def search_documents(query: str, document: str = "") -> tuple[str, list[dict]]:
    """Semantic search over the knowledge base. Returns the most relevant text
    passages with their document name and page number.
    Args:
      query: a short, specific search query (use keywords, not a long question).
      document: optional. A document name to search only inside that document.
        Leave empty to search all documents.
    If results are weak or empty, rewrite the query (try rewrite_query or different
    keywords) and search again. For multi-part questions, search once per part."""
    store = get_store()
    source = None
    if document:
        source = resolve_document(document)
        if source is None:
            return not_found_message(document), []

    results = store.search(query, k=config.TOP_K, source=source)
    results = [r for r in results if r["score"] >= config.MIN_SCORE]
    if not results:
        return (
            "No relevant passages found for this query. Try different keywords, "
            "use rewrite_query, or check list_documents.",
            [],
        )
    return format_results(results), to_sources(results)


REWRITE_SYSTEM = """You convert user questions into search queries for a semantic \
document search engine.
Rules:
- Output 1 to 3 short, keyword-rich queries, ONE PER LINE, nothing else.
- If the question has several parts, write one query per part.
- Replace vague words and pronouns with concrete terms. Keep names, numbers, dates.
- No numbering, no explanations, no quotes."""


@tool
def rewrite_query(question: str) -> str:
    """Rewrite a vague, long, or multi-part question into 1-3 focused search
    queries. Use it BEFORE search_documents when the question has several parts or
    unclear wording, or AFTER a search that returned weak or irrelevant results.
    Args:
      question: the user's question (or the search query that failed)."""
    reply = get_llm(fast=True).invoke([
        ("system", REWRITE_SYSTEM),
        ("human", question),
    ])
    queries = []
    for line in str(reply.content).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if line:
            queries.append(line)
    queries = queries[:3] or [question]
    return "Suggested search queries:\n" + "\n".join(
        f"{i}. {q}" for i, q in enumerate(queries, start=1)
    )