"""Парсеры внешних каталогов."""

from vsitemssrag.scraping.vistasport import (
    DEFAULT_CATALOG_URL,
    HttpClient,
    collect_product_urls,
    parse_catalog_page,
    parse_product_page,
)

__all__ = [
    "DEFAULT_CATALOG_URL",
    "HttpClient",
    "collect_product_urls",
    "parse_catalog_page",
    "parse_product_page",
]
