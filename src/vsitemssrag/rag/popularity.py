"""Редактируемые приоритеты PO; это не статистика продаж или свойство товара."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vsitemssrag.rag.materials import material_facts

REGISTRY_VERSION = 1
NICHE_PENALTY = 0.25


@dataclass(frozen=True)
class Bestseller:
    name: str
    category: str
    popularity_boost: float
    aliases: tuple[str, ...] = ()


BESTSELLERS = (
    Bestseller("DONIC Appelgren Allplay", "control", 0.25),
    Bestseller("STIGA Allround Classic", "control", 0.25),
    Bestseller("YASAKA Sweden Extra", "control", 0.25),
    # В каталоге Primorac имеет тип OFF: приоритет не меняет его характеристики.
    Bestseller("BUTTERFLY Primorac", "control", 0.15, ("BUTTERFLY Zoran Primorac",)),
    Bestseller("BUTTERFLY Petr Korbel", "attack", 0.20),
    Bestseller("STIGA Clipper Wood", "attack", 0.20),
    Bestseller("BUTTERFLY Viscaria", "attack", 0.20),
    Bestseller("BUTTERFLY Timo Boll ALC", "attack", 0.20),
    Bestseller("STIGA Cybershape Carbon", "attack", 0.20),
)

_CONTROL = re.compile(r"\b(?:нович\w*|начина\w*|контрол\w*|all(?:round)?\b|beginner\w*)", re.I)
_BEGINNER = re.compile(r"\b(?:нович\w*|начина\w*|beginner\w*)", re.I)
_ATTACK = re.compile(r"\b(?:атак\w*|напад\w*|off\b|быстр\w*|скорост\w*|attack\w*)", re.I)
_DEFENSE = re.compile(r"\b(?:защит\w*|def\b|длинн\w*\s+шип\w*|long\s+pips?)", re.I)
_MATERIAL = re.compile(r"\b(?:карбон\w*|композит\w*|carbon|alc|zlc|inner\w*|outer\w*|внутрен\w*)", re.I)
_NO_COMPOSITE = re.compile(r"без\s+(?:композит\w*|карбон\w*|carbon)", re.I)
_SPECIAL_HANDLE = re.compile(r"\b(?:dotec|дотек|пробк\w*|анатом\w*|эргоном\w*|изогнут\w*)", re.I)
_HANDLE_SUFFIX = re.compile(r"\s+(?:fl|st|an|conc|concave|straight|anatomic|jp|made in japan)$")


def model_name(value: object) -> str:
    value = str(value or "").casefold().replace("ё", "е")
    value = re.sub(r"[^\w]+", " ", value).strip()
    value = re.sub(r"^(?:основание|blade)\s+", "", value)
    # Только варианты ручки/страны. Carbon, Super ALC, SK7, Senso и CWT
    # не удаляются: это другие модели, их нельзя бустить по подстроке.
    previous = None
    while value != previous:
        previous, value = value, _HANDLE_SUFFIX.sub("", value)
    return value


def bestseller(product: Mapping[str, Any]) -> Bestseller | None:
    name = model_name(product.get("name"))
    return next((entry for entry in BESTSELLERS
                 if any(name == model_name(alias) for alias in (entry.name, *entry.aliases))), None)


def _requested(pattern: re.Pattern[str], question: str) -> bool:
    for match in pattern.finditer(question):
        prefix = question[max(0, match.start() - 30):match.start()]
        if not re.search(r"\b(?:не|без|кроме|исключ\w*)\b[^,.;!?]*$", prefix, re.I):
            return True
    return False


def wants_special_handle(question: str) -> bool:
    return _requested(_SPECIAL_HANDLE, question)


def ranking_modifiers(question: str, product: Mapping[str, Any]) -> tuple[float, float]:
    """Буст категории запроса и штраф нишевой ручке для общих запросов контроля."""
    defense = _requested(_DEFENSE, question)
    attack = _requested(_ATTACK, question)
    control = _requested(_BEGINNER, question) or (_requested(_CONTROL, question) and not attack)
    special_handle = wants_special_handle(question)
    entry = bestseller(product)
    boost = 0.0
    # Явный запрос Dotec не должен теряться за популярными альтернативами.
    explicit_dotec = _requested(re.compile(r"\b(?:dotec|дотек)\b", re.I), question)
    if entry and not defense and not explicit_dotec:
        category = "control" if control else "attack" if attack else None
        if category is None or entry.category == category:
            boost = entry.popularity_boost
    attrs = product.get("characteristics")
    attrs = attrs if isinstance(attrs, Mapping) else {}
    # Популярность не должна перекрывать явный запрос карбона/позиции слоя.
    if boost and _MATERIAL.search(question):
        construction = str(attrs.get("Конструкция") or "")
        composite = not _NO_COMPOSITE.search(construction) and bool(
            re.search(r"с\s+композит|carbon|карбон|\balc\b|\bzlc\b", construction + " " + str(product.get("name") or ""), re.I)
        )
        if _NO_COMPOSITE.search(question):
            if not _NO_COMPOSITE.search(construction):
                boost = 0.0
        elif not composite:
            boost = 0.0
        layouts = {f['material'] for f in material_facts(product) if f['role'] == 'composite_layout'}
        if re.search(r"\binner\w*|внутрен", question, re.I) and 'inner' not in layouts:
            boost = 0.0
        if re.search(r"\bouter\w*", question, re.I) and 'outer' not in layouts:
            boost = 0.0
    handle = " ".join(str(value) for key, value in attrs.items() if "ручк" in str(key).casefold())
    niche = bool(re.search(r"\b(?:dotec|дотек)\b", str(product.get("name") or ""), re.I))
    niche = niche or bool(re.search(r"пробк|изогнут", handle, re.I))
    penalty = NICHE_PENALTY if niche and control and not defense and not special_handle else 0.0
    return boost, penalty
