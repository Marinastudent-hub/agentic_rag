"""Helpers shared by all tools."""
import re
from difflib import get_close_matches
from pathlib import Path

from app.vectorstore.store import VectorStore

_store: VectorStore | None = None


def get_store() -> VectorStore:
    """One shared VectorStore for the whole app (tools and, later, the API)."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def _norm(text: str) -> str:
    return re.sub(r"[\s_\-.]+", "", text.lower())


def document_names() -> list[str]:
    return [d["source"] for d in get_store().list_documents()]


def resolve_document(name: str) -> str | None:
    """Match a loose name like 'Q3 report' to a real file like 'q3_report.txt'."""
    target = _norm(name or "")
    if not target:
        return None
    names = document_names()
    full = {_norm(n): n for n in names}
    stems = {_norm(Path(n).stem): n for n in names}

    if target in full:
        return full[target]
    if target in stems:
        return stems[target]
    for stem, original in stems.items():
        if len(target) >= 3 and (target in stem or stem in target):
            return original
    close = get_close_matches(target, list(stems), n=1, cutoff=0.6)
    return stems[close[0]] if close else None


def not_found_message(name: str) -> str:
    names = document_names()
    if not names:
        return "The knowledge base is empty. No documents have been uploaded."
    return f"Document '{name}' not found. Available documents: " + ", ".join(names)


def format_results(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"[{i}] Source: {r['source']} | Page: {r['page']} | "
            f"Relevance: {r['score']}\n{r['text']}"
        )
    return "\n\n".join(blocks)


def to_sources(results: list[dict]) -> list[dict]:
    return [{"source": r["source"], "page": r["page"], "score": r.get("score")}
            for r in results]