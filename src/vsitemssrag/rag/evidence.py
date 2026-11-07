"""Загрузка подтверждённых агрегатов отзывов для CLI, бота и eval."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

def load_community_evidence(path: Path | None) -> dict[int, dict[str, Any]]:
    """Загрузить только автоматически принятые или вручную проверенные match."""
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_products = payload.get("products") if isinstance(payload, Mapping) else None
    if not isinstance(raw_products, list):
        raise ValueError("evidence-файл не содержит список products")

    evidence_by_id: dict[int, dict[str, Any]] = {}
    for product in raw_products:
        if not isinstance(product, Mapping):
            continue
        match = product.get("match")
        if not isinstance(match, Mapping) or match.get("status") != "matched":
            continue
        evidence = match.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        try:
            product_id = int(product["product_id"])
        except (KeyError, TypeError, ValueError):
            continue
        evidence_by_id[product_id] = dict(evidence)
    return evidence_by_id


def attach_community_evidence(
    products: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for product in products:
        item = dict(product)
        try:
            evidence = evidence_by_id.get(int(product.get("product_id")))
        except (TypeError, ValueError):
            evidence = None
        if evidence is not None:
            item["community_evidence"] = dict(evidence)
        enriched.append(item)
    return enriched
