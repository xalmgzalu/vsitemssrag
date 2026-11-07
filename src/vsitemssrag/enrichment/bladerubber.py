"""Ограниченный импорт агрегатов публичных отзывов BladeRubber.net."""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from vsitemssrag.embeddings.documents import normalize_text
from vsitemssrag.enrichment.tabletennisdaily import (
    ExternalCatalogItem,
    ParsedReview,
    ParsedReviewPage,
    aggregate_review_evidence,
    classify_long_pips_relation,
    detect_signals,
)


SOURCE_NAME = "BladeRubber.net"
SOURCE_BASE_URL = "https://bladerubber.net"
SITEMAP_URL = f"{SOURCE_BASE_URL}/sitemap.xml"
ROBOTS_URL = f"{SOURCE_BASE_URL}/robots.txt"
USER_AGENT = "VistaSportRAGResearch/0.1 (limited 20-product pilot)"
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)
_PRODUCT_PATH = re.compile(r"^/blades/[^/]+/[^/]+$")


def parse_sitemap(xml: str) -> list[str]:
    root = ET.fromstring(xml)
    return [
        normalize_text(node.text)
        for node in root.iter()
        if node.tag.endswith("loc") and normalize_text(node.text)
    ]


def catalog_from_sitemap_urls(urls: list[str]) -> list[ExternalCatalogItem]:
    items: list[ExternalCatalogItem] = []
    for url in urls:
        parsed = urlsplit(url)
        if parsed.netloc not in {"bladerubber.net", "www.bladerubber.net"}:
            continue
        path = parsed.path.rstrip("/")
        if not _PRODUCT_PATH.fullmatch(path):
            continue
        slug = path.rsplit("/", 1)[-1]
        items.append(
            ExternalCatalogItem(
                name=slug.replace("-", " "),
                url=f"https://bladerubber.net{path}",
                rating=None,
                review_count=0,
            )
        )
    return items


def _product_json_ld(soup: BeautifulSoup) -> Mapping[str, Any]:
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(script.string or "null")
        except (TypeError, ValueError):
            continue
        candidates = payload if isinstance(payload, list) else [payload]
        for candidate in candidates:
            if isinstance(candidate, Mapping) and candidate.get("@type") == "Product":
                return candidate
    raise ValueError("BladeRubber.net не вернул Product JSON-LD")


def parse_product_page(
    html: str,
    *,
    page_url: str,
    max_reviews: int = 20,
) -> ParsedReviewPage:
    if max_reviews <= 0:
        raise ValueError("max_reviews должен быть больше нуля")
    soup = BeautifulSoup(html, "html.parser")
    product = _product_json_ld(soup)
    raw_brand = product.get("brand")
    brand = (
        normalize_text(raw_brand.get("name"))
        if isinstance(raw_brand, Mapping)
        else normalize_text(raw_brand)
    )
    name = normalize_text(product.get("name"))
    if brand and name.casefold().startswith(brand.casefold() + " "):
        name = name[len(brand) :].strip()

    aggregate = product.get("aggregateRating")
    rating: float | None = None
    if isinstance(aggregate, Mapping):
        try:
            rating = float(aggregate.get("ratingValue"))
        except (TypeError, ValueError):
            rating = None

    raw_reviews = product.get("review")
    if not isinstance(raw_reviews, list):
        raw_reviews = []
    parsed_reviews: list[ParsedReview] = []
    source_reviews_url = page_url.rstrip("/").removesuffix("/reviews") + "/reviews"
    for raw_review in raw_reviews[:max_reviews]:
        if not isinstance(raw_review, Mapping):
            continue
        body = normalize_text(raw_review.get("reviewBody"))
        raw_review_rating = raw_review.get("reviewRating")
        review_rating: float | None = None
        if isinstance(raw_review_rating, Mapping):
            try:
                review_rating = float(raw_review_rating.get("ratingValue"))
            except (TypeError, ValueError):
                review_rating = None
        parsed_reviews.append(
            ParsedReview(
                url=source_reviews_url,
                rating=review_rating,
                stats={},
                signals=detect_signals(body),
                relations=(
                    {"long_pips": relation}
                    if (relation := classify_long_pips_relation(body))
                    else {}
                ),
            )
        )

    return ParsedReviewPage(
        name=name,
        brand=brand,
        rating=rating,
        review_count=len(raw_reviews),
        reviews=tuple(parsed_reviews),
    )


def aggregate_bladerubber_evidence(
    page: ParsedReviewPage,
    *,
    source_url: str,
) -> dict[str, Any]:
    evidence = aggregate_review_evidence(
        page,
        source_url=source_url,
        source_name=SOURCE_NAME,
        terms_url=None,
    )
    evidence["robots_url"] = ROBOTS_URL
    return evidence


class BladeRubberClient:
    def __init__(
        self,
        *,
        delay: float = 1.5,
        timeout: float = 30.0,
        max_reviews: int = 20,
        session: requests.Session | None = None,
    ) -> None:
        if delay < 0 or timeout <= 0 or max_reviews <= 0:
            raise ValueError("некорректные параметры BladeRubberClient")
        self.delay = delay
        self.timeout = timeout
        self.max_reviews = max_reviews
        self.last_request_finished_at: float | None = None
        self.session = session or requests.Session()
        self._owns_session = session is None
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        if self._owns_session:
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
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "BladeRubberClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _get(self, url: str) -> str:
        if self.last_request_finished_at is not None:
            elapsed = time.monotonic() - self.last_request_finished_at
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
        try:
            response = self.session.get(url, timeout=(10, self.timeout))
            response.raise_for_status()
            return response.text
        finally:
            self.last_request_finished_at = time.monotonic()

    def fetch_catalog(self, *, max_pages: int = 1) -> list[ExternalCatalogItem]:
        del max_pages
        index_urls = parse_sitemap(self._get(SITEMAP_URL))
        sitemap_urls: list[str] = []
        for url in index_urls:
            if url.endswith(".xml"):
                sitemap_urls.extend(parse_sitemap(self._get(url)))
            else:
                sitemap_urls.append(url)
        return catalog_from_sitemap_urls(sitemap_urls)

    def fetch_reviews(self, item: ExternalCatalogItem) -> ParsedReviewPage:
        reviews_url = item.url.rstrip("/") + "/reviews"
        return parse_product_page(
            self._get(reviews_url),
            page_url=reviews_url,
            max_reviews=self.max_reviews,
        )
