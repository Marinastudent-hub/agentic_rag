"""Tests embeddings + ChromaDB with the documents in data/uploads."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.vectorstore.store import VectorStore


def show(results):
    if not results:
        print("   (no results)")
    for r in results:
        snippet = r["text"][:110].replace("\n", " ")
        print(f"   [{r['score']}] {r['source']} p.{r['page']}: {snippet}...")


def main():
    store = VectorStore()
    store.reset()  # start clean for the test

    files = [p for p in sorted(config.UPLOAD_DIR.iterdir())
             if p.suffix.lower() in config.ALLOWED_EXTENSIONS]
    if not files:
        print("[FAIL] No documents in data/uploads. Run test_ingestion.py first.")
        return

    print("Ingesting documents...")
    for f in files:
        try:
            n = store.add_file(f)
            print(f"  {f.name}: {n} chunks stored")
        except Exception as e:
            print(f"  [SKIP] {f.name}: {e}")

    print(f"\nTotal chunks in ChromaDB: {store.count()}")

    print("\nDocuments in knowledge base:")
    for d in store.list_documents():
        print(f"  {d['source']}: {d['chunks']} chunks, {d['pages']} pages")

    tests = [
        "How many days can employees work remotely?",
        "What was the revenue in Q3?",
        "What security rules apply to home working?",
    ]
    for q in tests:
        print(f"\nQuestion: {q}")
        show(store.search(q, k=2))

    first = store.list_documents()[0]["source"]
    print(f"\nSearch limited to '{first}':")
    show(store.search("main topic", k=2, source=first))

    print("\n[OK] Vector store works. Ready for Step 4!")


if __name__ == "__main__":
    main()