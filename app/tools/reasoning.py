
"""Document reasoning tools: summarization and comparison."""

from langchain_core.tools import tool

from app.llm import get_llm
from app.tools.shared import (
    get_store,
    resolve_document,
    not_found_message,
    format_results,
    to_sources,
)


MAX_CHUNKS = 18
BATCH_CHARS = 6000


def _sample(chunks, limit=MAX_CHUNKS):
    """Select representative chunks without repeating the same chunk."""
    if len(chunks) <= limit:
        return chunks

    if limit <= 1:
        return [chunks[0]]

    indexes = [
        round(i * (len(chunks) - 1) / (limit - 1))
        for i in range(limit)
    ]

    return [chunks[i] for i in indexes]


def _chunk_text(chunk):
    """Extract readable text from a stored chunk."""
    if isinstance(chunk, str):
        return chunk

    if isinstance(chunk, dict):
        return (
            chunk.get("text")
            or chunk.get("content")
            or chunk.get("page_content")
            or ""
        )

    return getattr(chunk, "page_content", str(chunk))


def _chunk_metadata(chunk):
    if isinstance(chunk, dict):
        return chunk.get("metadata", chunk)

    return getattr(chunk, "metadata", {}) or {}


def _source_records(chunks, document_name):
    records = []
    seen = set()

    for chunk in chunks:
        metadata = _chunk_metadata(chunk)

        source = metadata.get("source", document_name)
        page = metadata.get("page", "Unknown")

        key = (str(source), str(page))

        if key not in seen:
            seen.add(key)
            records.append({
                "source": str(source),
                "page": str(page),
            })

    return records


def _make_batches(chunks):
    batches = []
    current = []
    current_length = 0

    for chunk in chunks:
        text = _chunk_text(chunk).strip()

        if not text:
            continue

        if current and current_length + len(text) > BATCH_CHARS:
            batches.append(current)
            current = []
            current_length = 0

        current.append(text)
        current_length += len(text)

    if current:
        batches.append(current)

    return batches


@tool(response_format="content_and_artifact")
def summarize_document(document: str, focus: str = ""):
    """
    Summarize an uploaded document.

    Use this when the user requests a document summary or asks
    for the main ideas of a specific uploaded file.
    """
    document = (document or "").strip()
    focus = (focus or "").strip()

    if not document:
        return "Please provide a document filename.", []

    store = get_store()
    resolved = resolve_document(document)

    if not resolved:
        return not_found_message(document), []

    chunks = store.get_document_chunks(resolved)

    if not chunks:
        return (
            f"No indexed content was found in {resolved}.",
            [],
        )

    selected_chunks = _sample(chunks)
    batches = _make_batches(selected_chunks)

    if not batches:
        return (
            f"No readable text was found in {resolved}.",
            [],
        )

    llm = get_llm(fast=True)
    partial_summaries = []

    for index, batch in enumerate(batches, start=1):
        text = "\n\n".join(batch)

        prompt = f"""
Summarize only the information contained in the document excerpts.

Do not add outside facts.
Do not invent figures, names, dates, or conclusions.
If information is unclear or missing, say so.
Focus: {focus or "Main ideas and important information"}

Document: {resolved}
Excerpt group: {index}

EXCERPTS:
{text}
"""

        response = llm.invoke(prompt)
        partial_summaries.append(str(response.content))

    final_prompt = f"""
Create a clear, organized summary using only these partial summaries.

Do not add facts that are not present in them.
Do not invent quotations or page numbers.
Mention important limitations if the available excerpts are incomplete.

Document: {resolved}
Focus: {focus or "Main ideas and important information"}

PARTIAL SUMMARIES:
{chr(10).join(partial_summaries)}
"""

    response = get_llm().invoke(final_prompt)

    sources = _source_records(selected_chunks, resolved)

    source_lines = [
        f"[{item['source']}, p. {item['page']}]"
        for item in sources
    ]

    answer = str(response.content).strip()

    if source_lines:
        answer += "\n\nSources used:\n" + "\n".join(source_lines)

    return answer, sources


@tool(response_format="content_and_artifact")
def compare_documents(
    topic: str,
    document_a: str,
    document_b: str,
):
    """
    Compare two uploaded documents on a specified topic.

    Search both documents and compare only the retrieved passages.
    """
    topic = (topic or "").strip()
    document_a = (document_a or "").strip()
    document_b = (document_b or "").strip()

    if not topic or not document_a or not document_b:
        return (
            "Please provide a comparison topic and both document filenames.",
            [],
        )

    store = get_store()

    resolved_a = resolve_document(document_a)
    resolved_b = resolve_document(document_b)

    if not resolved_a:
        return not_found_message(document_a), []

    if not resolved_b:
        return not_found_message(document_b), []

    results_a = store.search(
        query=topic,
        k=4,
        source=resolved_a,
    )

    results_b = store.search(
        query=topic,
        k=4,
        source=resolved_b,
    )

    if not results_a or not results_b:
        return (
            "I could not find enough relevant passages in both documents "
            "to make a reliable comparison.",
            [],
        )

    excerpts_a = format_results(results_a)
    excerpts_b = format_results(results_b)

    prompt = f"""
Compare the two documents on the requested topic.

Use ONLY the supplied excerpts.
Do not use outside knowledge.
Do not assume that missing information means the opposite is true.
Clearly distinguish:
- Similarities
- Differences
- Information found in only one document
- Information that cannot be determined

Do not invent quotations, facts, or page numbers.
Cite claims using the source names and page numbers shown in the excerpts.

TOPIC:
{topic}

DOCUMENT A: {resolved_a}
{excerpts_a}

DOCUMENT B: {resolved_b}
{excerpts_b}
"""

    response = get_llm().invoke(prompt)

    sources = (
        to_sources(results_a)
        + to_sources(results_b)
    )

    unique_sources = []
    seen = set()

    for item in sources:
        key = (
            str(item.get("source")),
            str(item.get("page")),
        )

        if key not in seen:
            seen.add(key)
            unique_sources.append(item)

    return str(response.content).strip(), unique_sources
