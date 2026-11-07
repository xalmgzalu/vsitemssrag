"""Детерминированная валидация чисел evidence и опасных выводов."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from vsitemssrag.rag.context import community_evidence_lines


VALIDATOR_VERSION = 7


_CITATION_GROUP = re.compile(r"\[((?:\d+\s*,\s*)*\d+)]")
_EVIDENCE_PAIR = re.compile(r"(?<!\d)(\d+)\s+из\s+(\d+)(?!\d)", re.I)
_COMPATIBILITY_ASSERTION = (
    r"\b(?:подход[а-яё]*|совместим(?:о|а|ы|ый|ая|ое|ые|ым|ыми|ому)?|"
    r"подтвержд[а-яё]*|доказ[а-яё]*|рекоменд[а-яё]*|гарант[а-яё]*)\b"
)
_LONG_PIPS_OVERCLAIM = re.compile(
    rf"(?:{_COMPATIBILITY_ASSERTION}[^.!?\n]{{0,80}}длинн[а-я]*\s+шип|"
    rf"длинн[а-я]*\s+шип[^.!?\n]{{0,80}}{_COMPATIBILITY_ASSERTION})",
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
    r"(?:положительн[а-я]*\s+(?:направленн[а-я]*\s+)?опыт[^.!?\n]{0,100}"
    r"(?:не\s+зафиксирован|не\s+подтвержд[её]н|нет|отсутств)|"
    r"(?:нет|отсутств|недостаточно)[^.!?\n]{0,100}"
    r"(?:данн|подтвержд|сведен)[^.!?\n]{0,60}длинн[а-я]*\s+шип|"
    r"(?:данн|подтвержд|сведен)[^.!?\n]{0,100}длинн[а-я]*\s+шип"
    r"[^.!?\n]{0,60}(?:нет|отсутств|недостаточно))",
    re.I,
)
_SPEED_SUPERLATIVE = re.compile(
    r"\b(?:сам(?:ый|ая|ое|ые|ой|ую|ого)|наиболее|наибольш[а-яё]*|максимальн[а-яё]*)\s+"
    r"(?:(?:высок[а-яё]*|больш[а-яё]*|показател[а-яё]*|значени[а-яё]*|"
    r"заявленн[а-яё]*|паспортн[а-яё]*|уров[а-яё]*)\s+){0,3}"
    r"(?:быстр[а-яё]*|скорост[а-яё]*)",
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
    # Стабильный порядок диагностик валидатора не зависит от порядка вывода.
    return sorted(answer_citations(value))


def answer_citations(value: str) -> list[int]:
    """Уникальные ссылки в порядке первого появления, включая группы [2, 1]."""
    return list(dict.fromkeys(
        int(item.strip())
        for group in _CITATION_GROUP.findall(value)
        for item in group.split(",")
    ))


def has_unsupported_speed_superlative(answer: str) -> bool:
    return bool(_SPEED_SUPERLATIVE.search(answer))


def _long_pips_overclaim(answer: str) -> bool:
    # Снимается лишь отрицание подтверждения, а не утверждение «не подходит».
    # Остальная часть ответа остаётся под проверкой, включая фразы после «но».
    text = re.sub(
        r"совместимость[^.!?;\n]{0,70}длинн[а-я]*\s+шип[а-я]*"
        r"[^.!?;\n]{0,40}не\s+(?:подтвержд|доказ)[а-яё]*",
        "", answer, flags=re.I,
    )
    text = re.sub(r"\bне\s+(?:подтвержд|доказ|гарант)[а-яё]*", "", text, flags=re.I)
    text = re.sub(
        r"\b(?:нет|недостаточно)\s+(?:прямых\s+)?подтвержд[её]н[а-яё]*|"
        r"\bподтвержд[её]н[а-яё]*\s+нет",
        "", text, flags=re.I,
    )
    text = re.sub(
        r"\b(?:подтвержд[её]ни[а-яё]*|доказательств[а-яё]*)"
        r"(?=[^.!?;\n]{0,100}\bнет\b)",
        "", text, flags=re.I,
    )
    return bool(_LONG_PIPS_OVERCLAIM.search(text))


def validate_model_statistics(answer: str) -> list[str]:
    """Модель не формирует статистику даже когда случайно угадала числа."""
    without_refs = _CITATION_GROUP.sub("", answer)
    count_word = r"(?:один|одного|одном|два|двух|три|тр[её]х|четыр\w*|пять|пяти|шесть|шести|семь|семи|восемь|восьми|девять|девяти|десять|десяти)"
    review_noun = r"(?:отзыв|автор|пользовател|игрок)"
    for unit in re.split(r"[.!?\n]", without_refs):
        review = re.search(r"отзыв|сообществ|автор|пользовател|игрок", unit, re.I)
        numeric = re.search(
            rf"\b\d+\s*(?:%|процент|/\s*\d+|(?:просмотренн\w*\s+)?{review_noun})|"
            rf"\b{review_noun}\w*\s*(?:[:—-]|составляет)\s*\d+|"
            rf"\b{count_word}\s+(?:из\s+|(?:просмотренн\w*\s+)?{review_noun})|"
            r"большинств|половин|треть|четверт|кажд|все\s+отзыв",
            unit, re.I,
        )
        if _EVIDENCE_PAIR.search(unit) or (review and numeric):
            return ["модель сформировала статистику отзывов; числовой блок добавляет код"]
    return []


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
    require_evidence: bool = True,
) -> list[str]:
    errors: list[str] = []
    citations = _citations(answer)
    invalid = [ref for ref in citations if not 1 <= ref <= len(products)]
    if invalid:
        errors.append(f"несуществующие ссылки на товары: {invalid}")
    if len(citations) > 3:
        errors.append(f"упомянуто больше трёх товаров: {citations}")
    if question and _LONG_PIPS_QUERY.search(question):
        unconfirmed = []
        for ref in citations:
            if not 1 <= ref <= len(products):
                continue
            attrs = products[ref - 1].get("characteristics")
            category = attrs.get("Тип основания") if isinstance(attrs, Mapping) else None
            if str(category or "").strip().upper() not in {"DEF", "DEF+"}:
                unconfirmed.append(ref)
        if unconfirmed:
            errors.append(f"для long-pips нет явного каталожного типа DEF/DEF+: {unconfirmed}")
    if question and re.search(r"мягк|soft", question, re.I):
        if not re.search(r"мягк|soft", answer, re.I):
            errors.append("не разобран запрошенный критерий мягкости")
    if _INTERNAL_SCHEMA.search(answer):
        errors.append("в ответ попали внутренние английские имена evidence")
    if _long_pips_overclaim(answer):
        errors.append("упоминание длинных шипов превращено в совместимость")
    if has_unsupported_speed_superlative(answer):
        errors.append("неподтверждённая максимальная скорость без общей шкалы")
    if _UNSUPPORTED_INFERENCE.search(answer):
        errors.append("обнаружен неподтверждённый общий вывод о пригодности")

    allowed = allowed_community_evidence_lines(products)
    if require_evidence and question and _LONG_PIPS_QUERY.search(question):
        if not citations:
            errors.append("для конкретного long-pips запроса не предложено защитное основание")
        positive_lines = [
            line
            for line in allowed
            if line.startswith(
                "Положительный опыт использования с длинными шипами описан"
            )
            and any(ref in citations for ref in _citations(line))
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
