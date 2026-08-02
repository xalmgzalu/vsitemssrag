"""Извлечение доступных оснований из каталога VistaSport."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from vsitemssrag.models import Price, Product, ProductVariant


DEFAULT_CATALOG_URL = (
    "https://www.vistasport.ru/catalog/blades/filter/is_available-is-da/"
)
DEFAULT_USER_AGENT = (
    "vsitemssrag/0.1 (educational catalog parser; Python requests)"
)
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def iter_json_objects(value: Any) -> Iterator[dict[str, Any]]:
    """Обойти JSON-LD, включая массивы и контейнеры @graph."""
    if isinstance(value, list):
        for item in value:
            yield from iter_json_objects(item)
        return

    if not isinstance(value, dict):
        return

    yield value
    if "@graph" in value:
        yield from iter_json_objects(value["@graph"])


def schema_types(schema: dict[str, Any]) -> set[str]:
    return {str(item) for item in as_list(schema.get("@type"))}


def find_product_schema(soup: BeautifulSoup) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    for script in soup.select('script[type="application/ld+json"]'):
        raw_json = script.string or script.get_text()
        if not raw_json.strip():
            continue

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            continue

        for schema in iter_json_objects(parsed):
            if schema_types(schema) & {"ProductGroup", "Product"}:
                candidates.append(schema)

    if not candidates:
        raise ValueError("на странице не найден Product/ProductGroup JSON-LD")

    return next(
        (item for item in candidates if "ProductGroup" in schema_types(item)),
        candidates[0],
    )


def parse_price(value: Any) -> Price:
    if value in (None, ""):
        return None

    try:
        number = float(str(value).replace(" ", "").replace(",", "."))
    except ValueError:
        return None

    return int(number) if number.is_integer() else number


def parse_offer(product: dict[str, Any]) -> ProductVariant:
    offers = as_list(product.get("offers"))
    offer = next((item for item in offers if isinstance(item, dict)), {})
    availability = str(offer.get("availability", ""))

    return {
        "name": str(product["name"]) if product.get("name") else None,
        "sku": str(product["sku"]) if product.get("sku") else None,
        "price": parse_price(offer.get("price")),
        "currency": (
            str(offer["priceCurrency"]) if offer.get("priceCurrency") else None
        ),
        "available": availability.rstrip("/").endswith("InStock"),
    }


def extract_description(soup: BeautifulSoup, schema: dict[str, Any]) -> str:
    descriptions = [
        normalize_text(node.get_text(" ", strip=True))
        for node in soup.select('[itemprop="description"]')
    ]
    descriptions = [description for description in descriptions if description]

    if descriptions:
        return max(descriptions, key=len)

    return normalize_text(str(schema.get("description", "")))


def extract_characteristics(
    soup: BeautifulSoup, schema: dict[str, Any]
) -> dict[str, str]:
    characteristics: dict[str, str] = {}

    for item in as_list(schema.get("additionalProperty")):
        if not isinstance(item, dict):
            continue
        name = normalize_text(str(item.get("name", "")))
        value = normalize_text(str(item.get("value", "")))
        if name and value:
            characteristics[name] = value

    if characteristics:
        return characteristics

    table = soup.select_one(".product-specification table")
    if table is None:
        return characteristics

    for row in table.select("tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 2:
            continue
        name = next(cells[0].stripped_strings, "")
        value = normalize_text(cells[1].get_text(" ", strip=True))
        if name and value:
            characteristics[normalize_text(name)] = value

    return characteristics


def parse_product_page(html: str, source_url: str) -> Product:
    soup = BeautifulSoup(html, "html.parser")
    schema = find_product_schema(soup)

    brand_data = schema.get("brand")
    raw_brand = brand_data.get("name") if isinstance(brand_data, dict) else brand_data
    brand = str(raw_brand) if raw_brand else None

    variant_schemas = [
        item for item in as_list(schema.get("hasVariant")) if isinstance(item, dict)
    ]
    if not variant_schemas and "Product" in schema_types(schema):
        variant_schemas = [schema]

    variants = [parse_offer(variant) for variant in variant_schemas]
    priced_variants = [
        variant
        for variant in variants
        if variant["price"] is not None and variant["available"]
    ]
    if not priced_variants:
        priced_variants = [
            variant for variant in variants if variant["price"] is not None
        ]

    images = [str(image) for image in as_list(schema.get("image")) if image]
    characteristics = extract_characteristics(soup, schema)
    if brand and "Бренд" not in characteristics:
        characteristics = {"Бренд": brand, **characteristics}

    return {
        "name": normalize_text(str(schema.get("name", ""))),
        "link": str(schema.get("url") or source_url),
        "price": min(
            (variant["price"] for variant in priced_variants), default=None
        ),
        "currency": next(
            (
                variant["currency"]
                for variant in priced_variants
                if variant["currency"]
            ),
            None,
        ),
        "available": any(variant["available"] for variant in variants),
        "brand": brand,
        "description": extract_description(soup, schema),
        "characteristics": characteristics,
        "variants": variants,
        "image": images[0] if images else None,
    }


def parse_catalog_page(html: str, page_url: str) -> tuple[list[str], str | None]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()

    for anchor in soup.select("#catalog-content .products-flex-item a.name[href]"):
        url = urljoin(page_url, str(anchor["href"]))
        if url not in seen:
            seen.add(url)
            links.append(url)

    next_node = soup.select_one('link[rel~="next"][href], a[rel~="next"][href]')
    next_url = urljoin(page_url, str(next_node["href"])) if next_node else None
    return links, next_url


class HttpClient:
    def __init__(self, delay: float, timeout: float) -> None:
        self.delay = delay
        self.timeout = timeout
        self.last_request_finished_at: float | None = None
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.5",
            }
        )

        retries = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=1,
            status_forcelist=RETRYABLE_STATUS_CODES,
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def get_text(self, url: str) -> str:
        if self.last_request_finished_at is not None:
            elapsed = time.monotonic() - self.last_request_finished_at
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)

        try:
            response = self.session.get(
                url,
                timeout=(10, self.timeout),
                allow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        finally:
            self.last_request_finished_at = time.monotonic()


def collect_product_urls(
    client: HttpClient,
    catalog_url: str,
    limit: int,
    max_pages: int,
    on_page: Callable[[int, str], None] | None = None,
) -> list[str]:
    product_urls: list[str] = []
    seen_products: set[str] = set()
    seen_pages: set[str] = set()
    page_url: str | None = catalog_url
    page_number = 0

    while page_url and page_url not in seen_pages:
        if max_pages and page_number >= max_pages:
            break

        page_number += 1
        seen_pages.add(page_url)
        if on_page:
            on_page(page_number, page_url)
        links, next_url = parse_catalog_page(client.get_text(page_url), page_url)

        new_links = [url for url in links if url not in seen_products]
        if not new_links:
            break

        for url in new_links:
            seen_products.add(url)
            product_urls.append(url)
            if limit and len(product_urls) >= limit:
                return product_urls

        page_url = next_url

    return product_urls
