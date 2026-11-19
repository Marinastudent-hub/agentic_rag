
"""Split cleaned text into overlapping chunks."""
import re


def _split_long_text(text: str, size: int) -> list[str]:
    """Split oversized text while preferring word boundaries."""
    pieces = []

    while len(text) > size:
        split_at = text.rfind(" ", 0, size + 1)

        if split_at <= 0:
            split_at = size

        pieces.append(text[:split_at].strip())
        text = text[split_at:].strip()

    if text:
        pieces.append(text)

    return pieces


def _split_units(text: str, size: int) -> list[str]:
    """Split text into paragraphs, sentences, and smaller pieces."""
    units = []

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) <= size:
            units.append(paragraph)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) <= size:
                units.append(sentence)
            else:
                units.extend(_split_long_text(sentence, size))

    return units


def chunk_text(
    text: str,
    size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    """Create chunks that never exceed size characters."""
    if not text or not text.strip():
        return []

    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    units = _split_units(text, size)
    chunks = []
    current = ""

    for unit in units:
        if not current:
            current = unit
            continue

        candidate = current + "\n\n" + unit

        if len(candidate) <= size:
            current = candidate
            continue

        chunks.append(current)

        # Keep a bounded overlap from the previous chunk.
        tail = current[-overlap:].strip() if overlap else ""

        if tail and " " in tail:
            tail = tail[tail.find(" ") + 1:]

        available = size - len(unit) - 1

        if available > 0 and tail:
            tail = tail[-available:]
            if " " in tail:
                tail = tail[tail.find(" ") + 1:]

            current = (tail + " " + unit).strip()
        else:
            current = unit

    if current:
        chunks.append(current)

    # Final safety check: no chunk may exceed the configured limit.
    final_chunks = []

    for chunk in chunks:
        if len(chunk) <= size:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(_split_long_text(chunk, size))

    return [chunk for chunk in final_chunks if chunk]
