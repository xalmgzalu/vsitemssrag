"""Формирование и получение embeddings для каталога."""

from vsitemssrag.embeddings.documents import (
    build_product_document,
    build_search_query,
)
from vsitemssrag.embeddings.gemini import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    GeminiEmbeddingClient,
    GeminiEmbeddingError,
)

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_MODEL",
    "GeminiEmbeddingClient",
    "GeminiEmbeddingError",
    "build_product_document",
    "build_search_query",
]
