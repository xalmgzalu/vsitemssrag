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

_GENERIC_RECOMMENDATION_PATTERN = re.compile(
    r"\b(?:посовет|подбер|порекоменду|выбрать|нужн)[а-яё]*\b",
    flags=re.IGNORECASE,
)
_PRODUCT_PATTERN = re.compile(r"\bосновани[еяйю]?\b", flags=re.IGNORECASE)
_SPECIFICITY_PATTERN = re.compile(
    r"\b(?:нович|начина|любител|проф|бюджет|руб|₽|атак|напад|защит|"
    r"универс|контрол|скорост|быстр|медлен|карбон|carbon|композит|inner|"
    r"внутрен|мягк|ж[её]стк|шип|pips?|off|all|def|доник|donic|stiga|"
    r"butterfly|xiom|yasaka|nittaku|victas|joola|limba|лимб|koto|кото|"
    r"hinoki|хиноки|balsa|бальз|бальс|outer|dotec|дотек|пробк|анатом|эргоном)\w*\b",
    flags=re.IGNORECASE,
)

CLARIFYING_QUESTION = (
    "Чтобы подобрать основание без случайной рекомендации, уточните, пожалуйста: "
    "какой у вас уровень, стиль игры и бюджет?"
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


def requires_clarification(question: str) -> bool:
    """Распознать только явно общий запрос без единого критерия выбора."""
    normalized = " ".join(question.split())
    return bool(
        _GENERIC_RECOMMENDATION_PATTERN.search(normalized)
        and _PRODUCT_PATTERN.search(normalized)
        and not _SPECIFICITY_PATTERN.search(normalized)
        and extract_max_price_rub(normalized) is None
    )


def products_within_budget(
    products: Sequence[Mapping[str, Any]],
    max_price: Decimal,
    *,
    limit: int,
) -> list[Mapping[str, Any]]:
    """Оставить товары с известной ценой, не превышающей строгий бюджет."""
    if limit <= 0 or not max_price.is_finite() or max_price <= 0:
        raise ValueError("limit и конечный бюджет должны быть положительными")
    matches: list[Mapping[str, Any]] = []
    for product in products:
        raw_price = product.get("price")
        if raw_price is None:
            continue
        try:
            price = Decimal(str(raw_price))
        except InvalidOperation:
            continue
        if price.is_finite() and 0 < price <= max_price:
            matches.append(product)
        if len(matches) == limit:
            break
    return matches


def select_context_products(
    products: Sequence[Mapping[str, Any]],
    max_price: Decimal | None,
    *,
    limit: int,
) -> list[Mapping[str, Any]]:
    """Общий отбор для CLI и eval; пустой бюджетный результат не заменяется."""
    if max_price is not None:
        return products_within_budget(products, max_price, limit=limit)
    return list(products[:limit])


def no_budget_matches_answer(max_price: Decimal) -> str:
    amount = format(max_price, "f")
    if "." in amount:
        amount = amount.rstrip("0").rstrip(".")
    return (
        f"По вашему запросу не найдены варианты до {amount} ₽. "
        "Попробуйте изменить критерии поиска или увеличить бюджет."
    )
