"""Текстовые представления товаров и запросов для semantic search."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def build_product_document(product: Mapping[str, Any]) -> str:
    """Сформировать стабильный retrieval-документ без изменчивых полей."""
    name = normalize_text(product.get("name"))
    if not name:
        raise ValueError("у товара отсутствует name")

    fields = ["Категория: основание для настольного тенниса"]
    brand = normalize_text(product.get("brand"))
    if brand:
        fields.append(f"Бренд: {brand}")

    characteristics = product.get("characteristics") or {}
    for key, value in sorted(characteristics.items(), key=lambda item: item[0]):
        normalized_key = normalize_text(key)
        normalized_value = normalize_text(value)
        if not normalized_key or not normalized_value or normalized_key == "Бренд":
            continue
        fields.append(f"{normalized_key}: {normalized_value}")

    description = normalize_text(product.get("description"))
    if description and description.casefold() != name.casefold():
        fields.append(f"Описание: {description}")

    return f"title: {name} | text: " + " | ".join(fields)


def build_search_query(query: str) -> str:
    normalized = normalize_text(query)
    if not normalized:
        raise ValueError("поисковый запрос не может быть пустым")
    return f"task: search result | query: {normalized}"
