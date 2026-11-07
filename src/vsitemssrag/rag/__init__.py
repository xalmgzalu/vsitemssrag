"""Grounded-ответы по найденным товарам каталога."""

from vsitemssrag.rag.advisor import generate_advisor_answer
from vsitemssrag.rag.context import RAG_SYSTEM_INSTRUCTION, build_rag_prompt
from vsitemssrag.rag.constraints import (
    CLARIFYING_QUESTION,
    extract_max_price_rub,
    no_budget_matches_answer,
    products_within_budget,
    requires_clarification,
    select_context_products,
)
from vsitemssrag.rag.gemini import (
    GENERATION_MODEL,
    GeminiGenerationClient,
    GeminiGenerationError,
)

__all__ = [
    "GENERATION_MODEL",
    "RAG_SYSTEM_INSTRUCTION",
    "CLARIFYING_QUESTION",
    "GeminiGenerationClient",
    "GeminiGenerationError",
    "build_rag_prompt",
    "extract_max_price_rub",
    "generate_advisor_answer",
    "no_budget_matches_answer",
    "products_within_budget",
    "requires_clarification",
    "select_context_products",
]
