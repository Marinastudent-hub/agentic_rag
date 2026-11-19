
"""Shared helpers used by Agentic RAG tools."""

from difflib import get_close_matches

from app.vectorstore.store import VectorStore


_store = None


def get_store():
    """Return one reusable vector-store instance."""
    global _store

    if _store is None:
        _store = VectorStore()

    return _store


def _norm(value):
    return str(value or "").strip().lower().replace("\\", "/")


def document_names():
    """Return the uploaded document names."""
    documents = get_store().list_documents()

    names = []

    for item in documents:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = (
                item.get("source")
                or item.get("filename")
                or item.get("name")
            )
            if name:
                names.append(str(name))

    return sorted(set(names))


def resolve_document(name):
    """
    Resolve a document name safely.

    Prefer exact matches. Use fuzzy matching only when the result
    is sufficiently unambiguous.
    """
    name = str(name or "").strip()

    if not name:
        return None

    available = document_names()

    if not available:
        return None

    normalized = _norm(name)

    for document in available:
        if _norm(document) == normalized:
            return document

    basename_matches = [
        document
        for document in available
        if _norm(document).split("/")[-1]
        == normalized.split("/")[-1]
    ]

    if len(basename_matches) == 1:
        return basename_matches[0]

    if len(basename_matches) > 1:
        return None

    close_matches = get_close_matches(
        name,
        available,
        n=2,
        cutoff=0.88,
    )

    if len(close_matches) == 1:
        return close_matches[0]

    return None


def not_found_message(name):
    available = document_names()

    if not available:
        return "No documents have been uploaded yet."

    return (
        f"Could not identify the document '{name}'. "
        f"Available documents: {', '.join(available)}"
    )


def format_results(results):
    """Format retrieval results with their source and page metadata."""
    if not results:
        return "No relevant passages were found."

    lines = []

    for index, result in enumerate(results, start=1):
        if isinstance(result, dict):
            text = result.get("text") or result.get("content") or ""
            metadata = result.get("metadata", result)

            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")
            score = result.get("score")
        else:
            text = getattr(result, "page_content", str(result))
            metadata = getattr(result, "metadata", {}) or {}

            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")
            score = getattr(result, "score", None)

        header = (
            f"[{index}] Source: {source} | Page: {page}"
        )

        if score is not None:
            header += f" | Relevance: {float(score):.3f}"

        lines.append(f"{header}\n{text}")

    return "\n\n".join(lines)


def to_sources(results):
    """Convert search results into source metadata records."""
    sources = []
    seen = set()

    for result in results or []:
        if isinstance(result, dict):
            metadata = result.get("metadata", result)
            score = result.get("score")
        else:
            metadata = getattr(result, "metadata", {}) or {}
            score = getattr(result, "score", None)

        source = metadata.get("source")
        page = metadata.get("page")

        if source is None or page is None:
            continue

        key = (str(source), str(page))

        if key in seen:
            continue

        seen.add(key)

        item = {
            "source": str(source),
            "page": str(page),
        }

        if score is not None:
            item["score"] = float(score)

        sources.append(item)

    return sources
