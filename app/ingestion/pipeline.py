"""Full ingestion pipeline: extract -> clean -> chunk -> chunks with metadata."""
import hashlib
from pathlib import Path

from app import config
from app.ingestion.extractors import extract
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import chunk_text


def process_file(path: Path) -> list[dict]:
    """
    Returns a list of chunks:
    {
      "id": "a1b2c3d4_p2_c0",
      "text": "...",
      "metadata": {"source": "report.pdf", "page": 2,
                   "chunk_index": 0, "file_type": ".pdf"}
    }
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    doc_hash = hashlib.md5(path.name.encode("utf-8")).hexdigest()[:8]
    is_pdf = ext == ".pdf"
    chunks = []

    for page in extract(path):
        cleaned = clean_text(page["text"], join_lines=is_pdf)
        if not cleaned:
            continue
        for i, piece in enumerate(
            chunk_text(cleaned, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        ):
            chunks.append({
                "id": f"{doc_hash}_p{page['page']}_c{i}",
                "text": piece,
                "metadata": {
                    "source": path.name,
                    "page": page["page"],
                    "chunk_index": i,
                    "file_type": ext,
                },
            })
    return chunks


def process_directory(directory: Path = config.UPLOAD_DIR) -> list[dict]:
    all_chunks = []
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() in config.ALLOWED_EXTENSIONS:
            try:
                all_chunks.extend(process_file(path))
            except Exception as e:
                print(f"[SKIP] {path.name}: {e}")
    return all_chunks