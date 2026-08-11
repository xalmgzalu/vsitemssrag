"""Grounded-ответы по найденным товарам каталога."""

from vsitemssrag.rag.context import RAG_SYSTEM_INSTRUCTION, build_rag_prompt
from vsitemssrag.rag.constraints import extract_max_price_rub, products_within_budget
from vsitemssrag.rag.gemini import (
    GENERATION_MODEL,
    GeminiGenerationClient,
    GeminiGenerationError,
)

__all__ = [
    "GENERATION_MODEL",
    "RAG_SYSTEM_INSTRUCTION",
    "GeminiGenerationClient",
    "GeminiGenerationError",
    "build_rag_prompt",
    "extract_max_price_rub",
    "products_within_budget",
]
