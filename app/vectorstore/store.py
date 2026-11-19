
"""ChromaDB vector store: add, search, list, fetch and delete documents."""
from pathlib import Path

import chromadb

from app import config
from app.ingestion.pipeline import process_file
from app.vectorstore.embeddings import embed_texts, embed_query


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR)
        )
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict]) -> int:
        """Embed and upsert chunks in batches."""
        if not chunks:
            return 0

        batch_size = 100

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            texts = [item["text"] for item in batch]
            ids = [item["id"] for item in batch]
            metadatas = [item["metadata"] for item in batch]

            embeddings = embed_texts(texts)

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        return len(chunks)

    def add_file(self, path: Path) -> int:
        """Process and safely replace an existing document."""
        path = Path(path)

        # Process first. Existing indexed data remains untouched on failure.
        chunks = process_file(path)
        source = path.name

        old_data = self.collection.get(
            where={"source": source},
            include=["metadatas"],
        )
        old_ids = set(old_data.get("ids", []))
        new_ids = {chunk["id"] for chunk in chunks}

        # Upsert the new version before deleting obsolete chunks.
        added = self.add_chunks(chunks)

        stale_ids = list(old_ids - new_ids)

        if stale_ids:
            self.collection.delete(ids=stale_ids)

        return added

    def search(
        self,
        query: str,
        k: int = config.TOP_K,
        source: str | None = None,
    ) -> list[dict]:
        """Search for the most relevant document chunks."""
        total = self.collection.count()

        if total == 0:
            return []

        if not query or not query.strip():
            return []

        result_count = min(max(1, k), total)

        kwargs = {}
        if source:
            kwargs["where"] = {"source": source}

        results = self.collection.query(
            query_embeddings=[embed_query(query)],
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
            **kwargs,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches = []

        for text, metadata, distance in zip(
            documents, metadatas, distances
        ):
            metadata = metadata or {}
            matches.append({
                "text": text,
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", 1),
                "score": round(1 - float(distance), 4),
            })

        return matches

    def list_documents(self) -> list[dict]:
        """Return indexed document names and their chunk/page counts."""
        data = self.collection.get(include=["metadatas"])
        grouped = {}

        for metadata in data.get("metadatas", []):
            if not metadata:
                continue

            source = metadata.get("source")
            if not source:
                continue

            page = int(metadata.get("page", 1))

            if source not in grouped:
                grouped[source] = {
                    "source": source,
                    "chunks": 0,
                    "pages": 0,
                }

            grouped[source]["chunks"] += 1
            grouped[source]["pages"] = max(
                grouped[source]["pages"], page
            )

        return sorted(
            grouped.values(),
            key=lambda item: item["source"].lower(),
        )

    def get_document_chunks(self, source: str) -> list[dict]:
        """Fetch all chunks belonging to one document."""
        data = self.collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        chunks = []

        for text, metadata in zip(
            data.get("documents", []),
            data.get("metadatas", []),
        ):
            metadata = metadata or {}

            chunks.append({
                "text": text,
                "source": metadata.get("source", source),
                "page": int(metadata.get("page", 1)),
                "chunk_index": int(metadata.get("chunk_index", 0)),
            })

        return sorted(
            chunks,
            key=lambda item: (item["page"], item["chunk_index"]),
        )

    def delete_document(self, source: str) -> None:
        """Delete every indexed chunk for a document."""
        self.collection.delete(where={"source": source})

    def count(self) -> int:
        """Return total indexed chunk count."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(config.CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
