"""Безопасное HTML-представление и кнопки по проверенным ссылкам каталога."""
from __future__ import annotations

import html
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from urllib.parse import urlsplit

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from vsitemssrag.rag.context import COMMUNITY_EVIDENCE_HEADING, community_evidence_lines
from vsitemssrag.rag.service import AdviceResult
from vsitemssrag.rag.validation import answer_citations


_CITATION = re.compile(r"\[((?:\d+\s*,\s*)*\d+)]")
_RUBLE_PRICE = re.compile(
    r"(?<![\w.,])(?P<amount>\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)"
    r"(?:[.,](?P<cents>\d{1,2}))?\s*(?:RUB\b|руб(?:\.|\b)|₽)", re.I,
)
_LIST_PREFIX = re.compile(r"^\s*(?:(?:\d+[.)]|[1-9]\ufe0f?\u20e3|[-*•])\s+|#{1,6}\s*)")
_SCALE_CAVEAT = re.compile(r"шкал[а-яё]*", re.I)
_SCALE_FOOTNOTE = "ℹ️ Шкалы скорости и контроля производителей различаются; напрямую сравнивать их нельзя."


def format_price(value: object, currency: str = "RUB") -> str:
    """Рубли без копеек: округление half-up, неизвестная цена не становится нулём."""
    try:
        amount = Decimal(str(value).replace(" ", "").replace("\u00a0", "").replace("\u202f", "").replace(",", "."))
        if not amount.is_finite() or amount < 0:
            return "цена не указана"
        rounded = amount.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return "цена не указана"
    unit = "₽" if currency.upper() in {"RUB", "RUR", "РУБ", "РУБ.", "₽"} else currency
    return f"{rounded:,.0f} {unit}".replace(",", " ")


def format_prices(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        amount = match["amount"] + "." + (match["cents"] or "0")
        return format_price(amount)
    return _RUBLE_PRICE.sub(replace, text)


def _number(position: int) -> str:
    return f"{position}\ufe0f\u20e3" if position < 10 else f"{position}."


def _clean_text(text: str) -> str:
    text = format_prices(text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def render_answer(result: AdviceResult) -> str:
    """Карточки и клавиатура используют одну нумерацию, исходный RAG-текст не меняется.

    Каталожные поля оформляем сами. Пояснения переносим целыми строками,
    не пересказывая их и не отрезая отрицания. Сравнения с несколькими товарами
    сохраняют явные ссылки на новые номера. Доверенный блок отзывов собираем
    отдельно по тем же данным, только если он был в проверенном ответе.
    """
    cited = result.cited_products
    if not cited:
        return result.text
    positions = {ref: position for position, (ref, _) in enumerate(cited, 1)}
    notes: dict[int, list[str]] = {ref: [] for ref, _ in cited}
    products = dict(cited)
    intro: list[str] = []
    shared: list[str] = []
    footnotes: list[str] = []
    body, evidence_separator, _ = result.text.partition("\n\n" + COMMUNITY_EVIDENCE_HEADING)

    def display_refs(match: re.Match[str]) -> str:
        numbers = [positions[ref] for ref in answer_citations(match[0]) if ref in positions]
        if not numbers:
            return ""
        label = "вариант" if len(numbers) == 1 else "варианты"
        return f"({label} {', '.join(map(str, numbers))})"

    seen_product = False
    active_ref: int | None = None
    for raw in body.splitlines():
        if not raw.strip():
            active_ref = None
            continue
        line = _LIST_PREFIX.sub("", raw.strip())
        # Самостоятельная оговорка становится короткой сноской. Смешанные
        # предложения переносим целиком, сохраняя факты и ограничения.
        sentences = re.findall(r".+?(?:[.!?](?=\s|$)|$)", line)
        retained = []
        for sentence in sentences:
            if _SCALE_CAVEAT.search(sentence):
                without_refs = _CITATION.sub("", sentence)
                has_facts = bool(re.search(
                    r"\d|основан|модель|совместим|мягк|отзыв|шип|бюджет|цен", without_refs, re.I,
                )) or any(str(p.get("name") or "") in sentence for _, p in cited)
                footnote = (
                    "ℹ️ " + _clean_text(_CITATION.sub(display_refs, sentence))
                    if has_facts else _SCALE_FOOTNOTE
                )
                if footnote not in footnotes:
                    footnotes.append(footnote)
                continue
            retained.append(sentence)
        line = " ".join(part.strip() for part in retained)
        refs = [ref for ref in answer_citations(line) if ref in positions]
        if not line:
            continue
        if len(refs) == 1:
            ref = refs[0]
            clean = _CITATION.sub("", line).replace("**", "")
            clean = _clean_text(clean)
            # Название уже стоит в заголовке карточки; остальное сохраняется.
            name = str(products[ref].get("name") or "")
            for prefix in (name, name.removeprefix("Основание ")):
                if prefix and re.match(re.escape(prefix) + r"(?=$|[\s—–\-:.,])", clean, re.I):
                    clean = clean[len(prefix):].lstrip(" —–-:.,")
                    break
            if clean:
                notes[ref].append(clean)
            seen_product = True
            active_ref = ref
        else:
            clean = _clean_text(_CITATION.sub(display_refs, line))
            if not refs and active_ref is not None:
                # Продолжение карточки (например, цена/оговорка без повторной
                # цитаты) относится к её заголовку до пустой строки.
                notes[active_ref].append(clean)
            else:
                (shared if seen_product or refs else intro).append(clean)
                active_ref = None
            seen_product = seen_product or bool(refs)

    sections = ["\n".join(intro)] if intro else []
    for position, (ref, product) in enumerate(cited, 1):
        price = format_price(product.get("price"), str(product.get("currency") or "RUB"))
        card = [f"{_number(position)} **{product['name']} — {price}**"]
        attrs = product.get("characteristics")
        attrs = attrs if isinstance(attrs, Mapping) else {}
        kind = attrs.get("Тип основания")
        layers = attrs.get("Количество слоев") or attrs.get("Количество слоёв")
        type_parts = [str(kind)] if kind else []
        if layers:
            type_parts.append(f"{layers} слоёв")
        if type_parts:
            card.append("• Тип и слои: " + "; ".join(type_parts))
        construction = attrs.get("Конструкция")
        if construction:
            card.append(f"• Конструкция: {construction}")
        if notes[ref]:
            card.append("• Особенности: " + " ".join(dict.fromkeys(notes[ref])))
        if evidence_separator:
            evidence = community_evidence_lines(ref, product.get("community_evidence"))
            if evidence:
                card.append("• Отзывы сообщества:")
                card.extend("  — " + _clean_text(_CITATION.sub("", line)) for line in evidence)
                data = product.get("community_evidence")
                if isinstance(data, Mapping) and data.get("source_url"):
                    card.append(f"Источник отзывов: {data['source_url']}")
        sections.append("\n".join(card))
    if shared:
        sections.append("\n".join(shared))
    if evidence_separator:
        footnotes.append("ℹ️ " + COMMUNITY_EVIDENCE_HEADING)
    if footnotes:
        sections.append("\n".join(footnotes))
    return "\n\n".join(sections)


def split_text(text: str, limit: int = 3500) -> list[str]:
    """Ограничение в UTF-16 единицах; не режем emoji и не теряем текст."""
    if limit < 2:
        raise ValueError("limit должен быть не меньше 2")
    chunks: list[str] = []
    while text:
        units = 0
        cut = 0
        for char in text:
            width = 2 if ord(char) > 0xFFFF else 1
            if units + width > limit:
                break
            units += width
            cut += 1
        if cut < len(text):
            boundary = max(text.rfind("\n", 0, cut), text.rfind(" ", 0, cut))
            if boundary >= cut // 2:
                cut = boundary + 1
        chunks.append(text[:cut])
        text = text[cut:]
    return chunks or [""]


def telegram_html(text: str) -> str:
    # Все внешние теги сначала экранируются; поддерживается лишь простой bold.
    escaped = html.escape(text, quote=False)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)


def is_catalog_url(value: str) -> bool:
    try:
        url = urlsplit(value)
        return (
            url.scheme == "https" and url.hostname in {"vistasport.ru", "www.vistasport.ru"}
            and url.username is None and url.password is None and url.port in {None, 443}
            and url.path.startswith("/catalog/")
        )
    except ValueError:
        return False


def answer_keyboard(result: AdviceResult, response_id: str) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    cited = result.cited_products
    # Не выдаём частичную клавиатуру, в которой первая кнопка ведёт к карточке 2.
    if all(is_catalog_url(str(product.get("source_url") or "")) for _, product in cited):
        for position, (_, product) in enumerate(cited, 1):
            label = f"{_number(position)} {product['name']}"
            url = str(product["source_url"])
            rows.append([InlineKeyboardButton(text=label[:64], url=url)])
    rows.append([
        InlineKeyboardButton(text="👍", callback_data=f"fb:{response_id}:up"),
        InlineKeyboardButton(text="👎", callback_data=f"fb:{response_id}:down"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)
