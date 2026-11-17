"""ChromaDB vector store: add, search, list, fetch and delete documents."""
from pathlib import Path

import chromadb

from app import config
from app.ingestion.pipeline import process_file
from app.vectorstore.embeddings import embed_texts, embed_query


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- Adding ----------
    def add_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        batch = 100
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            texts = [c["text"] for c in part]
            self.collection.upsert(          # upsert = no duplicates on re-ingest
                ids=[c["id"] for c in part],
                documents=texts,
                embeddings=embed_texts(texts),
                metadatas=[c["metadata"] for c in part],
            )
        return len(chunks)

    def add_file(self, path: Path) -> int:
        """Ingest one file. Old chunks of the same file are replaced."""
        path = Path(path)
        self.delete_document(path.name)
        return self.add_chunks(process_file(path))

    # ---------- Searching ----------
    def search(self, query: str, k: int = config.TOP_K,
               source: str | None = None) -> list[dict]:
        """Semantic search. Optionally limit to one document."""
        if self.collection.count() == 0:
            return []
        kwargs = {}
        if source:
            kwargs["where"] = {"source": source}
        res = self.collection.query(
            query_embeddings=[embed_query(query)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
            **kwargs,
        )
        results = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append({
                "text": text,
                "source": meta["source"],
                "page": meta["page"],
                "score": round(1 - dist, 4),   # cosine similarity, higher = better
            })
        return results

    # ---------- Documents ----------
    def list_documents(self) -> list[dict]:
        data = self.collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in data["metadatas"]:
            d = docs.setdefault(
                meta["source"], {"source": meta["source"], "chunks": 0, "pages": 0}
            )
            d["chunks"] += 1
            d["pages"] = max(d["pages"], meta["page"])
        return sorted(docs.values(), key=lambda d: d["source"])

    def get_document_chunks(self, source: str) -> list[dict]:
        """All chunks of one document, in reading order."""
        data = self.collection.get(
            where={"source": source}, include=["documents", "metadatas"]
        )
        chunks = [
            {"text": t, "source": m["source"], "page": m["page"],
             "chunk_index": m["chunk_index"]}
            for t, m in zip(data["documents"], data["metadatas"])
        ]
        return sorted(chunks, key=lambda c: (c["page"], c["chunk_index"]))

    def delete_document(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(config.CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )