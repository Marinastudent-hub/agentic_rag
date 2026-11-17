"""Split text into overlapping chunks, respecting paragraph/sentence edges."""
import re


def _split_units(text: str, size: int) -> list[str]:
    units = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if len(sent) <= size:
                units.append(sent)
            else:  # very long sentence: hard split
                units.extend(sent[i:i + size] for i in range(0, len(sent), size))
    return units


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    chunks, current = [], ""
    for unit in _split_units(text, size):
        if not current:
            current = unit
        elif len(current) + 2 + len(unit) <= size:
            current += "\n\n" + unit
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if tail and " " in tail:                 # start overlap at a word edge
                tail = tail[tail.index(" ") + 1:]
            current = f"{tail} {unit}".strip() if tail else unit
    if current:
        chunks.append(current)
    return chunks