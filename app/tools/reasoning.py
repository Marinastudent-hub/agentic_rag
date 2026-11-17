"""Reasoning tools: summarize_document and compare_documents."""
from langchain_core.tools import tool

from app import config
from app.llm import get_llm
from app.tools.shared import (
    get_store, resolve_document, not_found_message, format_results, to_sources,
)

MAX_CHUNKS = 18       # keeps long documents within free-tier token limits
BATCH_CHARS = 6000


def _sample(chunks: list[dict], n: int) -> list[dict]:
    """Pick n chunks evenly spread across the document."""
    if len(chunks) <= n:
        return chunks
    step = (len(chunks) - 1) / (n - 1)
    return [chunks[round(i * step)] for i in range(n)]


def _batches(chunks: list[dict]) -> list[str]:
    batches, current, size = [], [], 0
    for c in chunks:
        line = f"[p.{c['page']}] {c['text']}"
        if current and size + len(line) > BATCH_CHARS:
            batches.append("\n\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        batches.append("\n\n".join(current))
    return batches


@tool(response_format="content_and_artifact")
def summarize_document(document: str, focus: str = "") -> tuple[str, list[dict]]:
    """Summarize one whole document. Use this for 'summarize', 'overview', or
    'what is this document about' requests. For specific facts, use
    search_documents instead.
    Args:
      document: the document name (e.g. 'q3_report.txt').
      focus: optional topic to emphasize in the summary."""
    store = get_store()
    name = resolve_document(document)
    if name is None:
        return not_found_message(document), []

    chunks = store.get_document_chunks(name)
    total = len(chunks)
    chunks = _sample(chunks, MAX_CHUNKS)
    note = f" (based on {len(chunks)} of {total} sections)" if len(chunks) < total else ""

    focus_line = f" Focus on: {focus}." if focus else ""
    map_prompt = (
        "Summarize the following document excerpts in a concise way. Keep key facts, "
        "numbers and names. Mention page numbers like (p.3) for important points."
        + focus_line
    )
    llm_fast, llm_main = get_llm(fast=True), get_llm()

    partials = [
        str(llm_fast.invoke([("system", map_prompt), ("human", b)]).content)
        for b in _batches(chunks)
    ]
    if len(partials) == 1:
        summary = partials[0]
    else:
        summary = str(llm_main.invoke([
            ("system",
             "Combine these partial summaries into one coherent summary. Keep key "
             "facts, numbers and page references like (p.3)." + focus_line),
            ("human", "\n\n---\n\n".join(partials)),
        ]).content)

    pages = sorted({c["page"] for c in chunks})
    sources = [{"source": name, "page": p, "score": None} for p in pages]
    return f"Summary of {name}{note}:\n\n{summary}", sources


@tool(response_format="content_and_artifact")
def compare_documents(topic: str, document_a: str, document_b: str) -> tuple[str, list[dict]]:
    """Compare two documents on a specific topic. Finds the relevant passages in
    each document and reports similarities and differences with citations.
    Args:
      topic: what to compare (e.g. 'remote work rules', 'revenue').
      document_a: name of the first document.
      document_b: name of the second document."""
    store = get_store()
    name_a, name_b = resolve_document(document_a), resolve_document(document_b)
    if name_a is None:
        return not_found_message(document_a), []
    if name_b is None:
        return not_found_message(document_b), []
    if name_a == name_b:
        return "Both names point to the same document. Choose two different documents.", []

    sections, sources = [], []
    for name in (name_a, name_b):
        hits = [r for r in store.search(topic, k=4, source=name)
                if r["score"] >= config.MIN_SCORE]
        if hits:
            sections.append(f"=== {name} ===\n{format_results(hits)}")
            sources.extend(to_sources(hits))
        else:
            sections.append(f"=== {name} ===\n(No relevant content found on this topic.)")

    reply = get_llm().invoke([
        ("system",
         "You compare two documents on a topic using ONLY the excerpts provided. "
         "Write: 1) Similarities, 2) Differences, 3) Anything missing from either "
         "document. Be concise. Cite every claim like (file.pdf, p.2). If a document "
         "has no relevant content, say so instead of guessing."),
        ("human", f"Topic: {topic}\n\n" + "\n\n".join(sections)),
    ])
    return str(reply.content), sources