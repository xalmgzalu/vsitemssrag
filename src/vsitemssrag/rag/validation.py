"""Детерминированная валидация чисел evidence и опасных выводов."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from vsitemssrag.rag.context import community_evidence_lines


VALIDATOR_VERSION = 2


_CITATION_GROUP = re.compile(r"\[((?:\d+\s*,\s*)*\d+)]")
_EVIDENCE_PAIR = re.compile(r"(?<!\d)(\d+)\s+из\s+(\d+)(?!\d)", re.I)
_LONG_PIPS_OVERCLAIM = re.compile(
    r"(?:(?:подход|совмест|подтверж|рекоменд)[^.!?\n]{0,80}длинн[а-я]*\s+шип|"
    r"длинн[а-я]*\s+шип[^.!?\n]{0,80}(?:подход|совмест|подтверж|рекоменд))",
    re.I,
)
_COMPARISON = re.compile(
    r"\b(?:выше|ниже|больше|меньше|быстрее|медленнее|против|"
    r"по сравнению|более\s+высок[а-я]*|наибольш[а-я]*|сам[а-я]*|"
    r"максимальн[а-я]*)\b",
    re.I,
)
_UNSUPPORTED_INFERENCE = re.compile(
    r"(?:традиционно[^.!?\n]{0,80}(?:нович|обуч)|"
    r"(?:идеальн|отлично\s+подход)[а-я]*[^.!?\n]{0,80}(?:нович|обуч)|"
    r"способству[а-я]*[^.!?\n]{0,50}(?:контрол|мягк)|"
    r"характерн[а-я]*[^.!?\n]{0,50}хорош[а-я]*\s+чувств|"
    r"подразумева[а-я]*[^.!?\n]{0,50}(?:контрол|мягк))",
    re.I,
)
_LONG_PIPS_QUERY = re.compile(r"длинн[а-я]*\s+шип|long\s+pips?", re.I)
_LONG_PIPS_DISCLAIMER = re.compile(
    r"(?:положительн[а-я]*\s+опыт[^.!?\n]{0,100}"
    r"(?:не\s+зафиксирован|не\s+подтвержд[её]н|нет|отсутств)|"
    r"(?:нет|отсутств|недостаточно)[^.!?\n]{0,100}"
    r"(?:данн|подтвержд|сведен)[^.!?\n]{0,60}длинн[а-я]*\s+шип|"
    r"(?:данн|подтвержд|сведен)[^.!?\n]{0,100}длинн[а-я]*\s+шип"
    r"[^.!?\n]{0,60}(?:нет|отсутств|недостаточно))",
    re.I,
)
_INTERNAL_SCHEMA = re.compile(
    r"\b(?:community_evidence|summary_lines|soft_feel|hard_or_stiff_feel|"
    r"defensive_play|long_pips)\b",
    re.I,
)


def _normalize(value: str) -> str:
    return " ".join(value.replace("*", "").replace("`", "").casefold().split())


def _citations(value: str) -> list[int]:
    return sorted(
        {
            int(item.strip())
            for group in _CITATION_GROUP.findall(value)
            for item in group.split(",")
        }
    )


def _evidence_semantics(value: str) -> set[str]:
    patterns = {
        "beginner": r"начинающ|нович",
        "control": r"контрол",
        "defensive_play": r"защитн[а-я]*\s+игр",
        "hard_or_stiff_feel": r"ж[её]стк|упруг",
        "soft_feel": r"мягк",
        "speed": r"скорост",
        "long_pips_positive": r"положительн[а-я]*\s+опыт[^.!?]*длинн[а-я]*\s+шип",
        "long_pips_negative": r"отрицательн[а-я]*\s+опыт[^.!?]*длинн[а-я]*\s+шип",
        "long_pips_unclear": r"длинн[а-я]*\s+шип[^.!?]*без\s+однозначн",
    }
    return {
        semantic
        for semantic, pattern in patterns.items()
        if re.search(pattern, value, re.I)
    }


def _allowed_numeric_facts(lines: Sequence[str]) -> set[tuple[int, str, str, str]]:
    facts: set[tuple[int, str, str, str]] = set()
    for line in lines:
        refs = _citations(line)
        pairs = _EVIDENCE_PAIR.findall(line)
        semantics = _evidence_semantics(line)
        if len(refs) != 1 or len(pairs) != 1 or len(semantics) != 1:
            continue
        numerator, denominator = pairs[0]
        facts.add((refs[0], next(iter(semantics)), numerator, denominator))
    return facts


def allowed_community_evidence_lines(
    products: Sequence[Mapping[str, Any]],
) -> list[str]:
    return [
        line
        for index, product in enumerate(products, start=1)
        for line in community_evidence_lines(
            index,
            product.get("community_evidence"),
        )
    ]


def validate_generated_answer(
    answer: str,
    products: Sequence[Mapping[str, Any]],
    *,
    question: str | None = None,
) -> list[str]:
    errors: list[str] = []
    citations = _citations(answer)
    invalid = [ref for ref in citations if not 1 <= ref <= len(products)]
    if invalid:
        errors.append(f"несуществующие ссылки на товары: {invalid}")
    if len(citations) > 3:
        errors.append(f"упомянуто больше трёх товаров: {citations}")
    if _INTERNAL_SCHEMA.search(answer):
        errors.append("в ответ попали внутренние английские имена evidence")
    if _LONG_PIPS_OVERCLAIM.search(answer):
        errors.append("упоминание длинных шипов превращено в совместимость")
    if _UNSUPPORTED_INFERENCE.search(answer):
        errors.append("обнаружен неподтверждённый общий вывод о пригодности")

    allowed = allowed_community_evidence_lines(products)
    if question and _LONG_PIPS_QUERY.search(question):
        if not citations:
            errors.append("для конкретного long-pips запроса не предложено защитное основание")
        positive_lines = [
            line
            for line in allowed
            if line.startswith(
                "Положительный опыт использования с длинными шипами описан"
            )
        ]
        if positive_lines and not any(
            _normalize(line).rstrip(".") in _normalize(answer)
            for line in positive_lines
        ):
            errors.append("не использована точная строка положительного long-pips evidence")
        if not positive_lines and not _LONG_PIPS_DISCLAIMER.search(answer):
            errors.append(
                "нет явной оговорки об отсутствии положительного long-pips evidence"
            )

    allowed_facts = _allowed_numeric_facts(allowed)
    answer_units = [
        sentence
        for line in answer.splitlines()
        for sentence in re.split(r"(?<=[.!?])\s+", line)
        if sentence.strip()
    ]
    for line in answer_units:
        pairs = _EVIDENCE_PAIR.findall(line)
        if not pairs:
            continue
        refs = _citations(line)
        semantics = _evidence_semantics(line)
        if len(pairs) != 1 or len(refs) != 1 or len(semantics) != 1:
            errors.append(
                "числовая evidence-фраза должна содержать один товар, один "
                "сигнал и одну пару N из M"
            )
            continue
        numerator, denominator = pairs[0]
        fact = (refs[0], next(iter(semantics)), numerator, denominator)
        if fact not in allowed_facts:
            errors.append(
                "числовая evidence-фраза не соответствует данным товара: "
                f"[{refs[0]}], {numerator} из {denominator}"
            )

    for line in answer.splitlines():
        refs = _citations(line)
        if len(refs) < 2 or not _COMPARISON.search(line) or not re.search(r"\d", line):
            continue
        if re.search(r"(?:цен|руб|rub|₽)", line, re.I):
            continue
        brands = {
            str(products[ref - 1].get("brand") or "").casefold()
            for ref in refs
            if 1 <= ref <= len(products)
        }
        brands.discard("")
        if len(brands) > 1:
            errors.append("прямое числовое сравнение товаров разных брендов")
    return list(dict.fromkeys(errors))
