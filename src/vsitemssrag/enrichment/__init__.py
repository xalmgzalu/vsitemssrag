"""Внешние источники доказательств для каталога."""

from vsitemssrag.enrichment.bladerubber import (
    BladeRubberClient,
    aggregate_bladerubber_evidence,
    catalog_from_sitemap_urls,
    parse_product_page,
    parse_sitemap,
)
from vsitemssrag.enrichment.tabletennisdaily import (
    TableTennisDailyClient,
    aggregate_review_evidence,
    exact_model_match,
    match_score,
    parse_catalog_page,
    parse_reviews_page,
    select_pilot_products,
)

__all__ = [
    "BladeRubberClient",
    "TableTennisDailyClient",
    "aggregate_bladerubber_evidence",
    "aggregate_review_evidence",
    "exact_model_match",
    "match_score",
    "catalog_from_sitemap_urls",
    "parse_catalog_page",
    "parse_product_page",
    "parse_reviews_page",
    "parse_sitemap",
    "select_pilot_products",
]
