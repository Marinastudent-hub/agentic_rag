
"""Clean extracted text."""
import re
import unicodedata


def clean_text(text: str, join_lines: bool = False) -> str:
    """
    Normalize extracted text.
    join_lines=True is intended for PDFs where line breaks
    usually represent visual wrapping rather than paragraphs.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Rejoin words split by a hyphen at the end of a line.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    if join_lines:
        # Preserve paragraph breaks, but join ordinary wrapped lines.
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize horizontal whitespace without removing paragraph breaks.
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
