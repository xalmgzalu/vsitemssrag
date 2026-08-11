"""Детерминированные ограничения, которые нельзя поручать генеративной модели."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any


_MAX_PRICE_PATTERN = re.compile(
    r"(?:до|не\s+дороже|максимум|бюджет(?:ом)?)\s*"
    r"(?P<amount>\d[\d\s\u00a0]*(?:[.,]\d+)?)\s*"
    r"(?P<thousands>тыс(?:яч(?:а|и|у)?)?\.?|[kк])?",
    flags=re.IGNORECASE,
)


def extract_max_price_rub(question: str) -> Decimal | None:
    """Извлечь простой верхний бюджет: «до 6000» или «до 6 тысяч»."""
    match = _MAX_PRICE_PATTERN.search(question)
    if not match:
        return None

    raw_amount = match.group("amount").replace(" ", "").replace("\u00a0", "")
    try:
        amount = Decimal(raw_amount.replace(",", "."))
    except InvalidOperation:
        return None
    if match.group("thousands"):
        amount *= 1000
    return amount if amount > 0 else None


def products_within_budget(
    products: Sequence[Mapping[str, Any]],
    max_price: Decimal,
    *,
    limit: int,
) -> list[Mapping[str, Any]]:
    """Оставить товары с известной ценой, не превышающей строгий бюджет."""
    matches: list[Mapping[str, Any]] = []
    for product in products:
        raw_price = product.get("price")
        if raw_price is None:
            continue
        try:
            price = Decimal(str(raw_price))
        except InvalidOperation:
            continue
        if price <= max_price:
            matches.append(product)
        if len(matches) == limit:
            break
    return matches
