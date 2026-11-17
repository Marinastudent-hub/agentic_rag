
"""Sentence Transformers embedding wrapper."""

from sentence_transformers import SentenceTransformer

from app import config

_model = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    global _model

    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)

    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert multiple texts into normalized vectors."""
    if not texts:
        return []

    vectors = get_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Convert one question into a vector."""
    return embed_texts([text])[0]
