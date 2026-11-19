
"""File ingestion pipeline."""
import hashlib
from pathlib import Path

from app import config
from app.ingestion.extractors import extract
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import chunk_text


def process_file(path: Path) -> list[dict]:
    """Extract, clean and chunk a single document."""
    path = Path(path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext not in config.ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    doc_hash = hashlib.md5(path.name.encode("utf-8")).hexdigest()[:8]
    is_pdf = ext == ".pdf"

    extracted_pages = extract(path)
    chunks = []

    for page in extracted_pages:
        cleaned = clean_text(
            page.get("text", ""),
            join_lines=is_pdf,
        )

        if not cleaned:
            continue

        page_number = int(page["page"])
        page_chunks = chunk_text(
            cleaned,
            size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )

        for chunk_index, piece in enumerate(page_chunks):
            chunks.append({
                "id": f"{doc_hash}_p{page_number}_c{chunk_index}",
                "text": piece,
                "metadata": {
                    "source": path.name,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "file_type": ext,
                },
            })

    if not chunks:
        raise ValueError("No readable text could be extracted from this file.")

    return chunks


def process_directory(directory: Path) -> list[dict]:
    """Process all supported files in a directory."""
    directory = Path(directory)
    all_chunks = []

    if not directory.exists():
        return all_chunks

    for path in directory.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
            continue

        try:
            all_chunks.extend(process_file(path))
        except Exception as exc:
            print(f"Skipped {path.name}: {exc}")

    return all_chunks
