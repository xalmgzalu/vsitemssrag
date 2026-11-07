"""Ограниченный исследовательский импорт агрегатов TableTennisDaily.

Полные тексты пользовательских отзывов намеренно не сохраняются. Pipeline
оставляет только числовые агрегаты, число упоминаний заранее заданных признаков
и URL исходных отзывов.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from statistics import fmean
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from vsitemssrag.embeddings.documents import normalize_text


SOURCE_NAME = "TableTennisDaily"
SOURCE_BASE_URL = "https://www.tabletennisdaily.com"
BLADES_CATALOG_URL = f"{SOURCE_BASE_URL}/equipment/categories/blades.3/"
TERMS_URL = f"{SOURCE_BASE_URL}/forum/help/terms/"
USER_AGENT = "VistaSportRAGResearch/0.1 (limited 20-product pilot)"
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


@dataclass(frozen=True)
class ExternalCatalogItem:
    name: str
    url: str
    rating: float | None
    review_count: int


@dataclass(frozen=True)
class ParsedReview:
    url: str
    rating: float | None
    stats: dict[str, float]
    signals: tuple[str, ...]
    relations: dict[str, str]


@dataclass(frozen=True)
class ParsedReviewPage:
    name: str
    brand: str
    rating: float | None
    review_count: int
    reviews: tuple[ParsedReview, ...]


_RATING_PATTERN = re.compile(r"(?P<rating>\d+(?:\.\d+)?)\s+star", re.I)
_INTEGER_PATTERN = re.compile(r"\d+")
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
_PRODUCT_PREFIX_PATTERN = re.compile(r"^основани[ея]\s+", re.I)

_BRAND_ALIASES = {
    "729": "friendship729",
    "friendship": "friendship729",
    "friendship729": "friendship729",
    "galaxy": "yinhe",
    "yinhegalaxy": "yinhe",
    "butterfly": "butterfly",
    "dhs": "dhs",
    "donic": "donic",
    "joola": "joola",
    "nittaku": "nittaku",
    "stiga": "stiga",
    "victas": "victas",
    "xiang": "xiang",
    "xiom": "xiom",
    "yasaka": "yasaka",
    "yinhe": "yinhe",
}

_SIGNAL_PATTERNS = {
    "soft_feel": re.compile(
        r"\b(?:soft|softer|softest|dwell|flex|flexible|flexibility)\b",
        re.I,
    ),
    "hard_or_stiff_feel": re.compile(
        r"\b(?:hard|harder|hardest|stiff|stiffer|stiffness|crisp)\b",
        re.I,
    ),
    "control": re.compile(
        r"\b(?:control|controlled|controllable|forgiving|placement)\b",
        re.I,
    ),
    "speed": re.compile(r"\b(?:fast|faster|fastest|speed|quick)\b", re.I),
    "defensive_play": re.compile(
        r"\b(?:defen[cs](?:e|ive|ively)?|defender|chop|chopper|chopping)\b",
        re.I,
    ),
    "beginner": re.compile(
        r"\b(?:beginner|beginners|novice|developing\s+player)\b",
        re.I,
    ),
}

_LONG_PIPS_PATTERN = re.compile(
    r"\b(?:long\s+(?:pips?|pimples?)|ox\s+(?:pips?|rubber)|pips?\s+ox)\b",
    re.I,
)
_LONG_PIPS_POSITIVE = re.compile(
    r"\b(?:use[ds]?|using|play(?:ed|ing)?|work(?:s|ed|ing)?|pair(?:ed|ing)?|"
    r"good|great|excellent|suitable|compatible|perfect|effective)\b",
    re.I,
)
_LONG_PIPS_NEGATIVE = re.compile(
    r"\b(?:not|never|avoid|bad|poor|unsuitable|incompatible|problem(?:atic)?|"
    r"difficult|worse|worst|doesn['’]?t|don['’]?t|can['’]?t)\b",
    re.I,
)


def _canonical_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _parse_rating(value: str | None) -> float | None:
    match = _RATING_PATTERN.search(value or "")
    return float(match.group("rating")) if match else None


def _parse_integer(value: str | None) -> int:
    match = _INTEGER_PATTERN.search(value or "")
    return int(match.group()) if match else 0


def parse_catalog_page(html: str, *, page_url: str = BLADES_CATALOG_URL) -> list[ExternalCatalogItem]:
    """Извлечь только карточки оснований, исключив featured reviews."""
    soup = BeautifulSoup(html, "html.parser")
    items: list[ExternalCatalogItem] = []
    seen: set[str] = set()

    for card in soup.select(".grid-item"):
        category = card.select_one(".grid-item-category-title-header a")
        if category is None or normalize_text(category.get_text()).casefold() != "blades":
            continue

        link = card.select_one("h3 a[href]")
        if link is None:
            continue
        name = normalize_text(link.get_text(" ", strip=True))
        url = _canonical_url(urljoin(page_url, str(link.get("href") or "")))
        if not name or not url or url in seen:
            continue

        rating_tag = card.select_one(".ratingStars[title]")
        reviews_link = card.select_one('a[href$="/reviews"]')
        items.append(
            ExternalCatalogItem(
                name=name,
                url=url,
                rating=_parse_rating(
                    str(rating_tag.get("title")) if rating_tag else None
                ),
                review_count=_parse_integer(
                    reviews_link.get_text(" ", strip=True) if reviews_link else None
                ),
            )
        )
        seen.add(url)
    return items


def _product_information(soup: BeautifulSoup) -> dict[str, str]:
    information: dict[str, str] = {}
    block = soup.select_one(".item-information--default")
    if block is None:
        return information
    for pair in block.select("dl.pairs"):
        key = pair.select_one("dt")
        value = pair.select_one("dd")
        if key and value:
            information[normalize_text(key.get_text(" ", strip=True))] = normalize_text(
                value.get_text(" ", strip=True)
            )
    return information


def _review_text(review: Any) -> str:
    parts: list[str] = []
    title = review.select_one(".review-title")
    if title:
        parts.append(title.get_text(" ", strip=True))
    for selector in (".rms-pros-container li", ".rms-cons-container li"):
        parts.extend(item.get_text(" ", strip=True) for item in review.select(selector))
    body = review.select_one("article.message-body .bbWrapper")
    if body:
        parts.append(body.get_text(" ", strip=True))
    return normalize_text(" ".join(parts))


def _review_stats(review: Any) -> dict[str, float]:
    stats: dict[str, float] = {}
    for field in review.select(".rms-review-fields dl"):
        name = field.select_one("dt")
        value = field.select_one(".bar-value")
        if not name or not value:
            continue
        try:
            parsed = float(normalize_text(value.get_text()))
        except ValueError:
            continue
        if math.isfinite(parsed):
            stats[normalize_text(name.get_text()).casefold()] = parsed
    return stats


def parse_reviews_page(html: str, *, page_url: str) -> ParsedReviewPage:
    """Разобрать отзывы, не возвращая и не сохраняя их полный текст."""
    soup = BeautifulSoup(html, "html.parser")
    information = _product_information(soup)
    heading = soup.select_one("h1")
    heading_text = normalize_text(heading.get_text(" ", strip=True) if heading else "")
    brand = information.get("Brand", "")
    name = heading_text
    if brand and name.casefold().startswith(brand.casefold() + " "):
        name = name[len(brand) :].strip()

    rating_tag = soup.select_one(".item-information--default .ratingStars[title]")
    parsed_reviews: list[ParsedReview] = []
    for review in soup.select(".rms-review"):
        text = _review_text(review)
        signals = detect_signals(text)
        link = review.select_one('a[href*="/review/"]')
        rating = review.select_one(".ratingStars[title]")
        parsed_reviews.append(
            ParsedReview(
                url=_canonical_url(
                    urljoin(page_url, str(link.get("href") or ""))
                    if link
                    else page_url
                ),
                rating=_parse_rating(str(rating.get("title")) if rating else None),
                stats=_review_stats(review),
                signals=signals,
                relations=_detect_relations(text),
            )
        )

    return ParsedReviewPage(
        name=name,
        brand=brand,
        rating=_parse_rating(str(rating_tag.get("title")) if rating_tag else None),
        review_count=_parse_integer(information.get("Reviews")),
        reviews=tuple(parsed_reviews),
    )


def detect_signals(text: str) -> tuple[str, ...]:
    return tuple(
        signal
        for signal, pattern in _SIGNAL_PATTERNS.items()
        if pattern.search(text)
    )


def classify_long_pips_relation(text: str) -> str | None:
    """Классифицировать только явно направленный контекст рядом с упоминанием."""
    relations: set[str] = set()
    for match in _LONG_PIPS_PATTERN.finditer(text):
        window = text[max(0, match.start() - 90) : match.end() + 90]
        if _LONG_PIPS_NEGATIVE.search(window):
            relations.add("negative")
        elif _LONG_PIPS_POSITIVE.search(window):
            relations.add("positive")
        else:
            relations.add("unclear")
    if not relations:
        return None
    return next(iter(relations)) if len(relations) == 1 else "unclear"


def _detect_relations(text: str) -> dict[str, str]:
    long_pips = classify_long_pips_relation(text)
    return {"long_pips": long_pips} if long_pips else {}


def aggregate_review_evidence(
    page: ParsedReviewPage,
    *,
    source_url: str,
    source_name: str = SOURCE_NAME,
    terms_url: str | None = TERMS_URL,
) -> dict[str, Any]:
    stat_values: dict[str, list[float]] = defaultdict(list)
    signal_urls: dict[str, list[str]] = defaultdict(list)
    relation_urls: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    review_ratings: list[float] = []

    for review in page.reviews:
        if review.rating is not None:
            review_ratings.append(review.rating)
        for name, value in review.stats.items():
            stat_values[name].append(value)
        for signal in review.signals:
            signal_urls[signal].append(review.url)
        for subject, relation in review.relations.items():
            relation_urls[subject][relation].append(review.url)

    stats = {
        name: {"mean": round(fmean(values), 3), "sample_size": len(values)}
        for name, values in sorted(stat_values.items())
    }
    mentions = {
        signal: {
            "review_count": len(urls),
            "review_urls": list(dict.fromkeys(urls))[:5],
        }
        for signal, urls in sorted(signal_urls.items())
    }
    relations = {
        subject: {
            "positive_review_count": len(values.get("positive", [])),
            "negative_review_count": len(values.get("negative", [])),
            "unclear_review_count": len(values.get("unclear", [])),
        }
        for subject, values in sorted(relation_urls.items())
    }
    return {
        "source": source_name,
        "source_url": source_url,
        "terms_url": terms_url,
        "evidence_kind": "subjective_community_aggregate",
        "rating": page.rating,
        "rating_count": page.review_count,
        "reviews_sampled": len(page.reviews),
        "sample_rating_mean": (
            round(fmean(review_ratings), 3) if review_ratings else None
        ),
        "stats": stats,
        "mentions": mentions,
        "relations": relations,
        "raw_review_text_retained": False,
    }


def _ascii_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_text(value)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    return _NON_ALNUM_PATTERN.sub("", normalized.casefold())


def normalize_brand(value: str | None) -> str:
    key = _ascii_key(value or "")
    return _BRAND_ALIASES.get(key, key)


def normalize_model(value: str, brand: str | None = None) -> str:
    normalized = _PRODUCT_PREFIX_PATTERN.sub("", normalize_text(value)).casefold()
    brand_key = normalize_brand(brand)
    tokens = re.findall(r"[a-zа-яё0-9]+", normalized, flags=re.I)
    if brand_key:
        tokens = [token for token in tokens if normalize_brand(token) != brand_key]
    return " ".join(tokens)


def match_score(product_name: str, product_brand: str | None, external_name: str) -> float:
    left = normalize_model(product_name, product_brand)
    right = normalize_model(external_name)
    if not left or not right:
        return 0.0
    if left.replace(" ", "") == right.replace(" ", ""):
        return 1.0
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, left, right).ratio()
    return round(0.55 * sequence + 0.45 * jaccard, 6)


def exact_model_match(
    product_name: str,
    product_brand: str | None,
    external_name: str,
) -> bool:
    left = normalize_model(product_name, product_brand).replace(" ", "")
    right = normalize_model(external_name).replace(" ", "")
    return bool(left and left == right)


def ranked_candidates(
    product: Mapping[str, Any],
    catalog: Sequence[ExternalCatalogItem],
    *,
    limit: int = 5,
) -> list[tuple[ExternalCatalogItem, float]]:
    ranked = [
        (
            item,
            match_score(
                str(product.get("name") or ""),
                str(product.get("brand") or ""),
                item.name,
            ),
        )
        for item in catalog
    ]
    ranked.sort(key=lambda pair: (-pair[1], pair[0].name.casefold(), pair[0].url))
    return ranked[:limit]


def select_pilot_products(
    products: Sequence[Mapping[str, Any]],
    targeted_urls: Sequence[str],
    *,
    limit: int = 20,
    seed: str = "ttd-pilot-v1",
) -> list[dict[str, Any]]:
    """Сначала взять целевые товары, остаток — по одному из крупнейших брендов."""
    if limit <= 0:
        raise ValueError("limit должен быть больше нуля")

    by_url = {str(product.get("source_url") or ""): product for product in products}
    selected: list[dict[str, Any]] = []
    selected_urls: set[str] = set()
    for source_url in targeted_urls:
        product = by_url.get(source_url)
        if product is None or source_url in selected_urls:
            continue
        selected.append({**product, "cohort": "targeted"})
        selected_urls.add(source_url)
        if len(selected) == limit:
            return selected

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for product in products:
        source_url = str(product.get("source_url") or "")
        if source_url and source_url not in selected_urls:
            groups[normalize_text(product.get("brand") or "Unknown")].append(product)

    brand_order = sorted(groups, key=lambda brand: (-len(groups[brand]), brand.casefold()))
    while len(selected) < limit:
        added = False
        for brand in brand_order:
            candidates = groups[brand]
            if not candidates:
                continue
            candidates.sort(
                key=lambda product: hashlib.sha256(
                    f"{seed}:{product.get('source_url', '')}".encode()
                ).hexdigest()
            )
            product = candidates.pop(0)
            selected.append({**product, "cohort": "coverage"})
            selected_urls.add(str(product.get("source_url") or ""))
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
    return selected


class TableTennisDailyClient:
    def __init__(
        self,
        *,
        delay: float = 1.5,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        if delay < 0 or timeout <= 0:
            raise ValueError("delay должен быть >= 0, timeout должен быть > 0")
        self.delay = delay
        self.timeout = timeout
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

    def __enter__(self) -> "TableTennisDailyClient":
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

    def fetch_catalog(self, *, max_pages: int = 11) -> list[ExternalCatalogItem]:
        if max_pages <= 0:
            raise ValueError("max_pages должен быть больше нуля")
        items: dict[str, ExternalCatalogItem] = {}
        for page in range(1, max_pages + 1):
            url = BLADES_CATALOG_URL if page == 1 else f"{BLADES_CATALOG_URL}?page={page}"
            parsed = parse_catalog_page(self._get(url), page_url=url)
            if not parsed:
                break
            for item in parsed:
                items[item.url] = item
        return list(items.values())

    def fetch_reviews(self, item: ExternalCatalogItem) -> ParsedReviewPage:
        url = item.url.rstrip("/") + "/reviews"
        return parse_reviews_page(self._get(url), page_url=url)


def serialize_catalog_item(item: ExternalCatalogItem) -> dict[str, Any]:
    return asdict(item)
