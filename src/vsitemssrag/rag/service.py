"""Общий синхронный RAG-сервис для CLI и асинхронных адаптеров."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
import requests

from vsitemssrag.embeddings import (
    EMBEDDING_MODEL, GeminiEmbeddingClient, GeminiEmbeddingError, build_search_query,
)
from vsitemssrag.rag.advisor import generate_advisor_answer
from vsitemssrag.rag.constraints import (
    CLARIFYING_QUESTION, extract_max_price_rub, no_budget_matches_answer,
    requires_clarification, select_context_products,
)
from vsitemssrag.rag.evidence import attach_community_evidence, load_community_evidence
from vsitemssrag.rag.gemini import GENERATION_MODEL, GeminiGenerationClient, GeminiGenerationError
from vsitemssrag.rag.validation import answer_citations
from vsitemssrag.settings import ConfigurationError, load_env, require_setting
from vsitemssrag.storage.postgres import Database

RAG_ERRORS = (
    ConfigurationError, GeminiEmbeddingError, GeminiGenerationError,
    psycopg.Error, requests.RequestException, ValueError, OSError,
)


@dataclass
class AdviceResult:
    text: str
    products: list[dict[str, Any]] = field(default_factory=list)
    origin: str = "gemini_validated"

    @property
    def cited_products(self) -> list[tuple[int, dict[str, Any]]]:
        return [
            (ref, self.products[ref - 1]) for ref in answer_citations(self.text)
            if 1 <= ref <= len(self.products)
        ]


def generation_model(model: str | None) -> str:
    load_env()
    return model or os.getenv("GEMINI_GENERATION_MODEL", "").strip() or GENERATION_MODEL


def ask_catalog(
    question: str, *, limit: int = 6, include_unavailable: bool = False,
    model: str | None = None, max_output_tokens: int = 1200,
    evidence_file: Path | None = None, timeout: float = 60.0,
) -> AdviceResult:
    if not question.strip() or limit <= 0 or timeout <= 0 or max_output_tokens <= 0:
        raise ValueError("вопрос и положительные параметры запроса обязательны")
    if requires_clarification(question):
        return AdviceResult(CLARIFYING_QUESTION, origin="deterministic_clarification")

    api_key = require_setting("GEMINI_API_KEY")
    selected_model = generation_model(model)
    evidence_by_id = load_community_evidence(evidence_file)
    max_price = extract_max_price_rub(question)
    with GeminiEmbeddingClient(api_key, delay=0, timeout=timeout) as client:
        query_embedding = client.embed(build_search_query(question))
    with Database.from_env() as database:
        database.apply_migrations()
        candidate_limit = min(max(limit * 5, 20), 100) if max_price else limit
        candidates = database.match_products(
            query_embedding, model=EMBEDDING_MODEL, limit=candidate_limit,
            only_available=not include_unavailable,
        )
    products = select_context_products(candidates, max_price, limit=limit)
    if not products:
        if max_price is not None:
            return AdviceResult(no_budget_matches_answer(max_price), origin="deterministic_no_budget_matches")
        return AdviceResult(
            "По вашему запросу не найдены подходящие основания. Попробуйте изменить критерии поиска.",
            origin="no_search_results",
        )
    enriched = attach_community_evidence(products, evidence_by_id)
    with GeminiGenerationClient(api_key, model=selected_model, timeout=timeout) as client:
        answer = generate_advisor_answer(
            client, question, enriched, max_output_tokens=max_output_tokens,
        )
    return AdviceResult(str(answer), enriched, getattr(answer, "origin", "gemini_validated"))
