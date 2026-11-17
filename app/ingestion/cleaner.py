"""Clean extracted text."""
import re
import unicodedata


def clean_text(text: str, join_lines: bool = False) -> str:
    """
    join_lines=True is for PDFs, where single line breaks are just
    visual wrapping and should become spaces.
    """
    text = unicodedata.normalize("NFKC", text)      # fixes ligatures like "ﬁ"
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)    # re-join hyphenated words
    if join_lines:
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)             # collapse spaces
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)          # max one blank line
    return text.strip()