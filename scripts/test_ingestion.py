"""Tests the ingestion pipeline on the files in data/uploads."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.ingestion.pipeline import process_directory

SAMPLES = {
    "company_policy.md": """# Remote Work Policy

Employees may work remotely up to 3 days per week with manager approval.
Remote employees must be reachable during core hours, 10:00 to 15:00.

## Equipment
The company provides a laptop and a monitor. A home office stipend of
$500 is paid once per year.

## Security
All remote work must use the company VPN. Public Wi-Fi without VPN is
not allowed.
""",
    "q3_report.txt": """Q3 Financial Summary

Revenue in Q3 was $4.2 million, up from $3.5 million in Q2.
Operating costs were $2.9 million in Q3 and $2.7 million in Q2.

The growth came mainly from the new subscription product, which added
$0.9 million in revenue. Customer churn fell from 5% to 3.8%.

Hiring: 12 new employees joined in Q3, mostly in engineering.
""",
}


def main():
    existing = [p for p in config.UPLOAD_DIR.iterdir()
                if p.suffix.lower() in config.ALLOWED_EXTENSIONS]
    if not existing:
        print("No documents found. Creating 2 sample files...\n")
        for name, content in SAMPLES.items():
            (config.UPLOAD_DIR / name).write_text(content, encoding="utf-8")

    chunks = process_directory()
    if not chunks:
        print("[FAIL] No chunks produced.")
        return

    per_doc = Counter(c["metadata"]["source"] for c in chunks)
    print(f"Total chunks: {len(chunks)}\n")
    for doc, n in per_doc.items():
        print(f"  {doc}: {n} chunks")

    sizes = [len(c["text"]) for c in chunks]
    print(f"\nChunk size  min={min(sizes)}  avg={sum(sizes)//len(sizes)}  max={max(sizes)}")

    print("\n--- Sample chunk ---")
    c = chunks[0]
    print("ID:      ", c["id"])
    print("Metadata:", c["metadata"])
    print("Text:    ", c["text"][:300])
    print("\n[OK] Ingestion works. Ready for Step 3!")


if __name__ == "__main__":
    main()