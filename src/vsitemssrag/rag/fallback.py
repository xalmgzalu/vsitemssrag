"""Резервный ответ из каталога без использования отклонённого текста модели."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from vsitemssrag.rag.context import community_evidence_lines, render_community_evidence


def build_catalog_fallback(
    question: str,
    products: Sequence[Mapping[str, Any]],
    references: Sequence[int],
) -> str:
    long_pips = bool(re.search(r"длинн[а-я]*\s+шип|long\s+pips?", question, re.I))
    carbon = bool(re.search(r"карбон|carbon|композит", question, re.I))
    inner = bool(re.search(r"inner|внутрен", question, re.I))
    selected: list[int] = []
    # Предпочитаем выбор модели, но берём только товары с явными критериями.
    for ref in dict.fromkeys([*references, *range(1, len(products) + 1)]):
        if not 1 <= ref <= len(products):
            continue
        product = products[ref - 1]
        attrs = product.get("characteristics") or {}
        if not isinstance(attrs, Mapping):
            attrs = {}
        if long_pips and str(attrs.get("Тип основания") or "").upper() not in {"DEF", "DEF+"}:
            continue
        name = str(product.get("name") or "")
        if carbon and not (
            str(attrs.get("Конструкция") or "").casefold() == "с композитом"
            or re.search(r"carbon|carbo|карбон|alc|zlc", name, re.I)
        ):
            continue
        if inner and not re.search(r"inner|внутрен", name + " " + str(attrs.get("Конструкция") or ""), re.I):
            continue
        selected.append(ref)
        if len(selected) == 3:
            break
    if not selected:
        return (
            "В поисковой выборке нет вариантов с явно подтверждёнными "
            "каталожными критериями запроса. Уточните требования к основанию?"
        )
    lines = ["По доступным данным каталога можно рассмотреть следующие варианты:"]
    for ref in selected:
        product = products[ref - 1]
        lines.append(f"\n[{ref}] {product['name']}")
        if product.get("price") is not None:
            lines.append(f"Цена: {product['price']} {product.get('currency') or 'RUB'} [{ref}].")
        attrs = product.get("characteristics") or {}
        if not isinstance(attrs, Mapping):
            attrs = {}
        for field in ("Тип основания", "Конструкция", "Количество слоев", "Толщина лопасти (мм)", "Контроль", "Скорость"):
            if attrs.get(field) is not None:
                lines.append(f"{field}: {attrs[field]} [{ref}].")
    if len({products[ref - 1].get("brand") for ref in selected}) > 1:
        lines.append("\nШкалы скорости и контроля производителей могут различаться; их нельзя напрямую сравнивать.")
    if long_pips:
        positive = any(
            line.startswith("Положительный опыт использования с длинными шипами описан")
            for ref in selected
            for line in community_evidence_lines(ref, products[ref - 1].get("community_evidence"))
        )
        lines.append(
            "\nОтдельный опыт не доказывает общую совместимость."
            if positive else
            "\nПоложительный опыт использования с длинными шипами в доступных данных не подтверждён."
        )
    if re.search(r"мягк|soft", question, re.I):
        direct_soft = any(
            re.search(r"мягк|soft", str(products[ref - 1].get("characteristics") or {}), re.I)
            for ref in selected
        )
        if not direct_soft:
            lines.append("\nПрямого подтверждения мягкости в приведённых каталожных характеристиках нет.")
    evidence = render_community_evidence(products, selected)
    if evidence:
        lines.append("\n" + evidence)
    return "\n".join(lines)
