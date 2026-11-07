"""Детерминированные проверки ответов для небольшого RAG A/B-эксперимента."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from vsitemssrag.rag.validation import validate_generated_answer, has_unsupported_speed_superlative
from vsitemssrag.rag.constraints import extract_max_price_rub, no_budget_matches_answer

GRADER_VERSION = 7


_CITATION_GROUP = re.compile(r"\[((?:\d+\s*,\s*)*\d+)]")
_INSUFFICIENT = re.compile(
    r"(?:(?:нет|недостаточно|отсутств(?:ует|уют)|не содерж(?:ит|ат)|"
    r"не указан[а-я]*)[^.!?\n]{0,80}(?:данн|информац|подтверж|характерист|"
    r"сведен|упоминан)|(?:данн|информац|подтверж|характерист|сведен|упоминан)"
    r"[^.!?\n]{0,100}(?:нет|отсутств(?:ует|уют)|не зафиксирован[а-я]*|"
    r"не указан[а-я]*|не содерж(?:ит|ат))|"
    r"положительн[а-я]*\s+опыт[^.!?\n]{0,100}"
    r"(?:не\s+зафиксирован|не\s+подтвержд[её]н|нет|отсутств))",
    re.IGNORECASE,
)
_CROSS_BRAND_CAVEAT = re.compile(
    r"(?:шкал[а-я]*[^.!?\n]{0,80}(?:различ|разн|несопостав)|"
    r"нельзя[^.!?\n]{0,80}(?:сравнива|сопоставля)|"
    r"не(?:льзя|корректно)[^.!?\n]{0,80}(?:сравнива|сопоставля)|"
    r"не\s+могут[^.!?\n]{0,40}(?:сравнив|сопостав))",
    re.IGNORECASE,
)


def answer_citations(answer: str) -> list[int]:
    return sorted(
        {
            int(value.strip())
            for group in _CITATION_GROUP.findall(answer)
            for value in group.split(",")
        }
    )


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def _product_text(product: Mapping[str, Any]) -> str:
    characteristics = product.get("characteristics")
    if not isinstance(characteristics, Mapping):
        characteristics = {}
    return " ".join(
        [
            str(product.get("name") or ""),
            *(f"{key} {value}" for key, value in characteristics.items()),
        ]
    ).casefold()


def _valid_evidence_pairs(
    products: Sequence[Mapping[str, Any]],
    signal: str,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for product in products:
        evidence = product.get("community_evidence")
        if not isinstance(evidence, Mapping):
            continue
        try:
            sampled = int(evidence.get("reviews_sampled") or 0)
        except (TypeError, ValueError):
            continue
        if signal == "long_pips":
            relations = evidence.get("relations")
            long_pips = (
                relations.get("long_pips")
                if isinstance(relations, Mapping)
                else None
            )
            raw_count = (
                long_pips.get("positive_review_count")
                if isinstance(long_pips, Mapping)
                else 0
            )
        else:
            mentions = evidence.get("mentions")
            mention = mentions.get(signal) if isinstance(mentions, Mapping) else None
            raw_count = (
                mention.get("review_count") if isinstance(mention, Mapping) else 0
            )
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        if 0 < count <= sampled:
            pairs.add((count, sampled))
    return pairs


def _uses_exact_aggregate(
    answer: str,
    products: Sequence[Mapping[str, Any]],
    signal: str,
) -> bool:
    if "отзыв" not in answer.casefold():
        return False
    return any(
        re.search(rf"(?<!\d){count}\s+из\s+{sampled}(?!\d)", answer)
        for count, sampled in _valid_evidence_pairs(products, signal)
    )


def _has_direct_characteristic(
    products: Sequence[Mapping[str, Any]],
    pattern: str,
) -> bool:
    expression = re.compile(pattern, re.IGNORECASE)
    return any(expression.search(_product_text(product)) for product in products)


def grade_answer(
    case: Mapping[str, Any],
    answer: str,
    products: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Оценить только заранее формализуемые свойства, без LLM-as-a-judge."""
    case_id = str(case.get("id") or "")
    max_price = extract_max_price_rub(str(case.get("question") or ""))
    if not products and max_price is not None:
        passed = answer.strip() == no_budget_matches_answer(max_price)
        return {
            "passed": passed,
            "citations": answer_citations(answer),
            "checks": [_check("no_budget_matches", passed, "честный ответ без рекомендаций")],
        }
    citations = answer_citations(answer)
    validation_question = str(case.get("question") or "")
    if case_id == "defense_long_pips" and not validation_question:
        validation_question = "длинные шипы"
    validation_errors = validate_generated_answer(
        answer,
        products,
        question=validation_question,
    )
    checks = [
        _check(
            "grounding_validator",
            not validation_errors,
            "; ".join(validation_errors) if validation_errors else "ошибок нет",
        ),
        _check(
            "valid_citations",
            all(1 <= citation <= len(products) for citation in citations),
            f"ссылки={citations}, товаров в контексте={len(products)}",
        ),
        _check(
            "at_most_three_products",
            len(citations) <= 3,
            f"уникальных ссылок на товары={len(citations)}",
        ),
    ]

    if case_id == "beginner_control_budget":
        max_price = Decimal(str(case["max_price_rub"]))
        cited_products = [products[index - 1] for index in citations if 1 <= index <= len(products)]
        prices: list[Decimal] = []
        for product in cited_products:
            try:
                prices.append(Decimal(str(product.get("price"))))
            except Exception:  # noqa: BLE001 - неизвестная цена должна провалить check
                prices.append(max_price + 1)
        checks.extend(
            [
                _check("recommends_product", bool(citations), f"ссылки={citations}"),
                _check(
                    "budget_respected",
                    bool(prices) and all(price <= max_price for price in prices),
                    f"цены={list(map(str, prices))}, бюджет={max_price}",
                ),
                _check(
                    "addresses_control",
                    "контрол" in answer.casefold(),
                    "ответ должен объяснять контроль",
                ),
            ]
        )

    elif case_id == "fast_attack_carbon":
        cited_products = [products[index - 1] for index in citations if 1 <= index <= len(products)]
        carbon = re.compile(
            r"(?:carbon|carbo|карбон|композит|alc|zlc|zylon|aramid)",
            re.IGNORECASE,
        )
        brands = {str(product.get("brand") or "").casefold() for product in cited_products}
        brands.discard("")
        checks.extend(
            [
                _check("recommends_product", bool(citations), f"ссылки={citations}"),
                _check(
                    "carbon_grounded",
                    bool(cited_products)
                    and all(carbon.search(_product_text(product)) for product in cited_products),
                    "каждый рекомендованный товар должен явно содержать carbon/карбон/composite-маркер",
                ),
                _check(
                    "cross_brand_scales_not_compared",
                    len(brands) < 2 or bool(_CROSS_BRAND_CAVEAT.search(answer)),
                    f"брендов={len(brands)}; при нескольких нужен явный caveat о шкалах",
                ),
                _check(
                    "no_unsupported_speed_superlative",
                    not has_unsupported_speed_superlative(answer),
                    "нельзя объявлять самый быстрый вариант без общей шкалы",
                ),
            ]
        )

    elif case_id == "defense_long_pips":
        exact_aggregate = _uses_exact_aggregate(answer, products, "long_pips")
        abstains = bool(_INSUFFICIENT.search(answer))
        evidence_available = bool(_valid_evidence_pairs(products, "long_pips"))
        checks.extend(
            [
                _check("recommends_product", bool(citations), f"ссылки={citations}"),
                _check(
                    "long_pips_claim_grounded_or_abstained",
                    exact_aggregate or abstains,
                    f"точный агрегат={exact_aggregate}, оговорка о нехватке данных={abstains}",
                ),
                _check(
                    "uses_available_long_pips_evidence",
                    not evidence_available or exact_aggregate,
                    f"evidence доступна={evidence_available}, использована={exact_aggregate}",
                ),
            ]
        )

    elif case_id == "soft_inner_carbon":
        exact_aggregate = _uses_exact_aggregate(answer, products, "soft_feel")
        direct_soft = _has_direct_characteristic(products, r"мягк|soft")
        abstains = bool(_INSUFFICIENT.search(answer))
        checks.extend(
            [
                _check("recommends_product", bool(citations), f"ссылки={citations}"),
                _check(
                    "addresses_soft_feel",
                    bool(re.search(r"мягк|soft", answer, re.IGNORECASE)),
                    "ответ должен разобрать запрошенное ощущение",
                ),
                _check(
                    "soft_feel_grounded_or_abstained",
                    exact_aggregate or direct_soft or abstains,
                    f"агрегат={exact_aggregate}, прямая характеристика={direct_soft}, оговорка={abstains}",
                ),
            ]
        )

    elif case_id == "underspecified_request":
        checks.extend(
            [
                _check(
                    "no_premature_recommendation",
                    not citations,
                    f"ссылки до уточнения={citations}",
                ),
                _check(
                    "one_clarifying_question",
                    answer.count("?") == 1,
                    f"вопросительных знаков={answer.count('?')}",
                ),
                _check(
                    "asks_relevant_constraint",
                    bool(re.search(r"уров|стил|бюджет|защит|атак", answer, re.IGNORECASE)),
                    "уточнение должно касаться уровня, стиля или бюджета",
                ),
            ]
        )
    else:
        checks.append(_check("known_case", False, f"неизвестный case id: {case_id}"))

    return {
        "passed": all(check["passed"] for check in checks),
        "citations": citations,
        "checks": checks,
    }
