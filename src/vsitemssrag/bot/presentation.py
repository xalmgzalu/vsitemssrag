"""Безопасное HTML-представление и кнопки по проверенным ссылкам каталога."""
from __future__ import annotations

import html
import re
from urllib.parse import urlsplit

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from vsitemssrag.rag.service import AdviceResult


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
    for ref, product in result.cited_products:
        url = str(product.get("source_url") or "")
        if is_catalog_url(url):
            label = f"[{ref}] {product['name']}"
            rows.append([InlineKeyboardButton(text=label[:64], url=url)])
    rows.append([
        InlineKeyboardButton(text="👍", callback_data=f"fb:{response_id}:up"),
        InlineKeyboardButton(text="👎", callback_data=f"fb:{response_id}:down"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)
