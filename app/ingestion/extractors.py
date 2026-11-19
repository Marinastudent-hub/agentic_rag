
"""Extract text from PDF, DOCX, TXT and MD files."""
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from pypdf import PdfReader


VIRTUAL_PAGE_CHARS = 3000


def _virtual_pages(text: str) -> list[dict]:
    """Split text files into virtual pages without losing long lines."""
    pages = []
    current = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = (
            [line[i:i + VIRTUAL_PAGE_CHARS]
             for i in range(0, len(line), VIRTUAL_PAGE_CHARS)]
            if len(line) > VIRTUAL_PAGE_CHARS
            else [line]
        )

        for part in parts:
            candidate = current + "\n" + part if current else part

            if len(candidate) > VIRTUAL_PAGE_CHARS and current:
                pages.append({
                    "page": len(pages) + 1,
                    "text": current,
                })
                current = part
            else:
                current = candidate

    if current:
        pages.append({
            "page": len(pages) + 1,
            "text": current,
        })

    return pages


def extract_pdf(path: Path) -> list[dict]:
    """Extract text page by page from a PDF."""
    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text,
            })

    if not pages:
        raise ValueError(
            "No selectable text found in this PDF. "
            "It may be scanned and require OCR."
        )

    return pages


def extract_docx(path: Path) -> list[dict]:
    """Extract DOCX paragraphs and tables in their original order."""
    document = Document(str(path))
    blocks = []

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()

            if text:
                blocks.append(text)

        elif isinstance(child, CT_Tbl):
            table = Table(child, document)

            for row in table.rows:
                cells = [
                    cell.text.strip().replace("\n", " ")
                    for cell in row.cells
                ]
                row_text = " | ".join(cell for cell in cells if cell)

                if row_text:
                    blocks.append(row_text)

    text = "\n\n".join(blocks)

    if not text.strip():
        raise ValueError("The DOCX file contains no readable text.")

    return _virtual_pages(text)


def extract_text_file(path: Path) -> list[dict]:
    """Read TXT or Markdown using common encodings."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    if not text.strip():
        raise ValueError("The file contains no readable text.")

    return _virtual_pages(text)


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".txt": extract_text_file,
    ".md": extract_text_file,
}


def extract(path: Path) -> list[dict]:
    """Extract text from a supported file."""
    path = Path(path)
    extractor = EXTRACTORS.get(path.suffix.lower())

    if extractor is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return extractor(path)
