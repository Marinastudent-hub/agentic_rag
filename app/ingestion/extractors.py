
"""Extract text from PDF, DOCX, TXT and MD files."""

from pathlib import Path

from pypdf import PdfReader
from docx import Document

VIRTUAL_PAGE_CHARS = 3000


def _virtual_pages(text: str) -> list[dict]:
    """Split files without real pages into virtual pages."""
    pages = []
    current = ""
    page_no = 1

    for para in text.split("\n"):
        if len(current) + len(para) + 1 > VIRTUAL_PAGE_CHARS and current.strip():
            pages.append({"page": page_no, "text": current})
            page_no += 1
            current = ""

        current += para + "\n"

    if current.strip():
        pages.append({"page": page_no, "text": current})

    return pages


def extract_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({"page": i, "text": text})

    if not pages:
        raise ValueError(
            f"No text found in {path.name}. It may be a scanned PDF (needs OCR)."
        )

    return pages


def extract_docx(path: Path) -> list[dict]:
    doc = Document(str(path))
    lines = [p.text for p in doc.paragraphs]

    for table in doc.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))

    return _virtual_pages("\n".join(lines))


def extract_text_file(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    return _virtual_pages(text)


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".txt": extract_text_file,
    ".md": extract_text_file,
}


def extract(path: Path) -> list[dict]:
    """Extract pages from a supported document."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext not in EXTRACTORS:
        raise ValueError(f"Unsupported file type: {ext}")

    return EXTRACTORS[ext](path)
