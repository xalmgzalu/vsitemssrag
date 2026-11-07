"""Онтология применяется только к явно указанным материалам/позициям слоёв."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files
from typing import Any


@lru_cache(maxsize=1)
def _ontology() -> dict[str, Any]:
    return json.loads(files("vsitemssrag.rag").joinpath("materials_ontology.json").read_text(encoding="utf-8"))


_ROLE_PREFIX = {
    "outer": r"(?:внешн[а-яё]*|верхн[а-яё]*|наружн[а-яё]*)\s+(?:сло[а-яё]*|шпон[а-яё]*)|outer\s+(?:pl(?:y|ies)|veneer)",
    "core": r"центральн[а-яё]*\s+сло[а-яё]*|сердцевин[а-яё]*|core(?:\s+ply)?",
}
_LAYOUT_FIELD = re.compile(r"конструкц|расположени[ея]\s+(?:композит|карбон)|composite.*(?:layout|placement)", re.I)
_COMPOSITION_FIELD = re.compile(r"состав|материал|composition|wood|veneer", re.I)
_NEGATED = re.compile(r"\b(?:не|без|нет|отсутств\w*)\b", re.I)
_UNCERTAIN = re.compile(r"\b(?:возмож\w*|предполож\w*|вероятн\w*|неизвест\w*|например|либо|или)\b|как у|\?", re.I)


def _wood_ids(text: str) -> list[str]:
    if _NEGATED.search(text) or _UNCERTAIN.search(text):
        return []
    return [key for key, value in _ontology()["materials"].items()
            if any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.I)
                   for alias in value.get("aliases", []))]


def _layout_ids(text: str) -> list[str]:
    if _NEGATED.search(text) or _UNCERTAIN.search(text):
        return []
    found = []
    for layout, pattern in (
        ("inner", r"\binner(?:fiber)?\b|внутренн[а-яё]*\s+(?:расположени[а-яё]*|карбон|композит)"),
        ("outer", r"\bouter(?:fiber)?\b|внешн[а-яё]*\s+(?:расположени[а-яё]*|карбон|композит)"),
    ):
        if re.search(pattern, text, re.I):
            found.append(layout)
    # Два варианта в сравнении не подтверждают конструкцию этого товара.
    return found if len(found) == 1 else []


def material_facts(product: Mapping[str, Any]) -> list[dict[str, str]]:
    """Каталожное доказательство отдельно от общих свойств материала.

    Название модели, число слоёв, слово «карбон» и текст отзывов не доказывают
    породу шпона либо положение композита. Структурные поля приоритетнее prose.
    """
    attrs = product.get("characteristics")
    attrs = attrs if isinstance(attrs, Mapping) else {}
    facts: list[dict[str, str]] = []

    def add(material: str, role: str, field: str, text: str) -> None:
        if not any(f["material"] == material and f["role"] == role for f in facts):
            facts.append({"material": material, "role": role, "catalog_field": field, "catalog_text": text})

    for field, raw in attrs.items():
        key, text = str(field), str(raw)
        if re.search(r"ручк|handle", key, re.I):
            continue
        role = next((role for role, pattern in _ROLE_PREFIX.items() if re.search(pattern, key, re.I)), None)
        if role or _COMPOSITION_FIELD.search(key):
            for material in _wood_ids(text):
                add(material, role or "unspecified", key, text)
        if _LAYOUT_FIELD.search(key):
            for layout in _layout_ids(text):
                add(layout, "composite_layout", key, text)

    description = str(product.get("description") or "")
    for role, prefix in _ROLE_PREFIX.items():
        if any(f["role"] == role for f in facts):
            continue
        for match in re.finditer(r"(?:" + prefix + r")\s*[:—-]?\s*([^,;().\n]{1,90})", description, re.I):
            preceding = re.split(r"[,.;()\n]", description[:match.start()])[-1][-35:]
            if _NEGATED.search(preceding) or re.search(r"в отличие от|например", preceding, re.I):
                continue
            material_text = re.split("(?:" + "|".join(_ROLE_PREFIX.values()) + ")", match[1], maxsplit=1, flags=re.I)[0]
            for material in _wood_ids(material_text):
                add(material, role, "description", description[match.start():match.start(1) + len(material_text)].strip())
    no_composite = any(re.search(r"без\s+(?:композит|карбон)", str(value), re.I)
                       for key, value in attrs.items() if _LAYOUT_FIELD.search(str(key)))
    if not no_composite and not any(f["role"] == "composite_layout" for f in facts):
        # В prose требуем явную фразу про структуру, чтобы название вроде
        # Innerforce в начале описания не становилось доказательством позиции.
        for match in re.finditer(
            r"(?:структур[а-яё]*|конструкц[а-яё]*|расположени[а-яё]*)[^.;\n]{0,70}|"
            r"(?:inner|outer)[-\s]+(?:структур[а-яё]*|конструкц[а-яё]*)[^.;\n]{0,50}|"
            r"сло[а-яё]*\s+(?:карбон[а-яё]*|композит[а-яё]*)[^.;\n]{0,100}"
            r"(?:внутренн|внешн)[а-яё]*\s+расположени[а-яё]*",
            description, re.I,
        ):
            for layout in _layout_ids(match[0]):
                add(layout, "composite_layout", "description", match[0])
    if no_composite or len({f["material"] for f in facts if f["role"] == "composite_layout"}) > 1:
        facts = [f for f in facts if f["role"] != "composite_layout"]
    return facts


def material_context(product: Mapping[str, Any]) -> dict[str, Any] | None:
    facts = material_facts(product)
    if not facts:
        return None
    ontology = _ontology()
    return {
        "kind": "material_reference_not_product_measurement",
        "version": ontology["schema_version"],
        "scope": ontology["scope"],
        "layers": [{**fact, **{key: value for key, value in ontology["materials"][fact["material"]].items()
                              if key != "aliases"}} for fact in facts],
    }
