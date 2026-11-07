"""Расширенный semantic retrieval → строгий бюджет → детерминированный ранг."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from vsitemssrag.rag.constraints import select_context_products
from vsitemssrag.rag.popularity import REGISTRY_VERSION, ranking_modifiers


RETRIEVAL_VERSION = "hybrid_popularity_v1"


def candidate_pool_limit(limit: int) -> int:
    if limit <= 0:
        raise ValueError("limit должен быть положительным")
    return min(max(limit * 5, 20), 100)


def _similarity(value: Any) -> float:
    try:
        score = float(value) if value is not None and not isinstance(value, bool) else -1.0
    except (TypeError, ValueError, OverflowError):
        return -1.0
    return score if math.isfinite(score) and -1 <= score <= 1 else -1.0


def select_ranked_products(
    question: str, candidates: Sequence[Mapping[str, Any]],
    max_price: Decimal | None, *, limit: int,
) -> list[dict[str, Any]]:
    """Бюджет применяется ко всему пулу ДО ранжирования, top-k — ПОСЛЕ.

    При равном score сохраняется исходный semantic-порядок. Входные товары
    не меняются; диагностика ranking не передаётся в prompt генератора.
    """
    if limit <= 0:
        raise ValueError("limit должен быть положительным")
    eligible = select_context_products(candidates, max_price, limit=max(len(candidates), 1))
    ranked = []
    for product in eligible:
        boost, penalty = ranking_modifiers(question, product)
        similarity = _similarity(product.get("similarity"))
        score = round(similarity + boost - penalty, 12)
        ranked.append({**product, "ranking": {
            "version": RETRIEVAL_VERSION, "registry_version": REGISTRY_VERSION,
            "cosine_similarity": similarity, "popularity_boost": boost,
            "niche_penalty": penalty, "final_score": score,
        }})
    return sorted(ranked, key=lambda item: item["ranking"]["final_score"], reverse=True)[:limit]
