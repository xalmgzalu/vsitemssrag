"""CLI ограниченного эксперимента с TableTennisDaily."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import psycopg
import requests

from vsitemssrag.enrichment.bladerubber import (
    ROBOTS_URL,
    SOURCE_NAME,
    BladeRubberClient,
    aggregate_bladerubber_evidence,
)
from vsitemssrag.enrichment.tabletennisdaily import (
    exact_model_match,
    normalize_brand,
    ranked_candidates,
    select_pilot_products,
    serialize_catalog_item,
)
from vsitemssrag.settings import ConfigurationError
from vsitemssrag.storage.postgres import Database, redact_secrets


DEFAULT_OUTPUT = Path("data/exports/review_enrichment_pilot_v2.json")
MIN_MATCH_SCORE = 0.58


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return parsed


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=positive, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-reviews", type=positive, default=20)
    parser.set_defaults(handler=run)


def _targeted_config() -> list[dict[str, str]]:
    resource = files("vsitemssrag.enrichment").joinpath("pilot_targets.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _overrides_config() -> dict[str, dict[str, str]]:
    resource = files("vsitemssrag.enrichment").joinpath(
        "pilot_match_overrides.json"
    )
    overrides = json.loads(resource.read_text(encoding="utf-8"))
    return {override["source_url"]: override for override in overrides}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _match_product(
    product: dict[str, Any],
    catalog: list[Any],
    client: BladeRubberClient,
    override: dict[str, str] | None = None,
) -> dict[str, Any]:
    if override is not None:
        item = next(
            (candidate for candidate in catalog if candidate.url == override["external_url"]),
            None,
        )
        if item is None:
            return {
                "status": "unmatched",
                "override_error": "external_url отсутствует в sitemap",
                "override": override,
                "inspected_candidates": [],
            }
        page = client.fetch_reviews(item)
        brand_matches = normalize_brand(page.brand) == normalize_brand(
            str(product.get("brand") or "")
        )
        if not brand_matches:
            return {
                "status": "unmatched",
                "override_error": "бренд override не совпал",
                "override": override,
                "inspected_candidates": [],
            }
        return {
            "status": "matched",
            "match_method": "verified_override",
            "score": 1.0,
            "score_margin": None,
            "external_name": page.name or item.name,
            "external_brand": page.brand,
            "external_url": item.url,
            "verification_url": override["verification_url"],
            "verification_reason": override["reason"],
            "evidence": aggregate_bladerubber_evidence(page, source_url=item.url),
            "inspected_candidates": [],
        }

    candidates = ranked_candidates(product, catalog, limit=5)
    inspected: list[dict[str, Any]] = []
    product_brand = normalize_brand(str(product.get("brand") or ""))
    best_family_candidate: dict[str, Any] | None = None

    for item, score in candidates:
        candidate = {**serialize_catalog_item(item), "score": score}
        if score < MIN_MATCH_SCORE:
            inspected.append(candidate)
            continue
        page = client.fetch_reviews(item)
        brand_matches = normalize_brand(page.brand) == product_brand
        inspected.append({**candidate, "external_brand": page.brand, "brand_matches": brand_matches})
        if not brand_matches:
            continue

        margin = score - (candidates[1][1] if len(candidates) > 1 else 0.0)
        result = {
            "status": "matched" if exact_model_match(
                str(product.get("name") or ""),
                str(product.get("brand") or ""),
                page.name or item.name,
            ) else "candidate",
            "match_method": "exact_normalized_model",
            "score": score,
            "score_margin": round(margin, 6),
            "external_name": page.name or item.name,
            "external_brand": page.brand,
            "external_url": item.url,
            "evidence": aggregate_bladerubber_evidence(page, source_url=item.url),
            "inspected_candidates": inspected,
        }
        if result["status"] == "matched":
            return result
        if best_family_candidate is None:
            best_family_candidate = result

    return best_family_candidate or {
        "status": "unmatched",
        "inspected_candidates": inspected,
    }


def _metrics(results: list[dict[str, Any]], catalog_count: int) -> dict[str, Any]:
    matched = [result for result in results if result["match"]["status"] == "matched"]
    candidates = [
        result for result in results if result["match"]["status"] == "candidate"
    ]
    signal_coverage: dict[str, int] = {}
    relation_coverage: dict[str, int] = {}
    for result in matched:
        mentions = result["match"]["evidence"]["mentions"]
        for signal in mentions:
            signal_coverage[signal] = signal_coverage.get(signal, 0) + 1
        relations = result["match"]["evidence"].get("relations") or {}
        for subject, relation in relations.items():
            for polarity in ("positive", "negative", "unclear"):
                if relation.get(f"{polarity}_review_count", 0):
                    key = f"{subject}_{polarity}"
                    relation_coverage[key] = relation_coverage.get(key, 0) + 1
    return {
        "external_catalog_items": catalog_count,
        "selected_products": len(results),
        "matched_products": len(matched),
        "candidate_products": len(candidates),
        "coverage_rate": round(len(matched) / len(results), 4) if results else 0.0,
        "matched_with_reviews": sum(
            result["match"]["evidence"]["reviews_sampled"] > 0 for result in matched
        ),
        "reviews_sampled": sum(
            result["match"]["evidence"]["reviews_sampled"] for result in matched
        ),
        "signal_product_coverage": dict(sorted(signal_coverage.items())),
        "relation_product_coverage": dict(sorted(relation_coverage.items())),
    }


def run(args: argparse.Namespace) -> int:
    if args.delay < 0 or args.timeout <= 0:
        print("Ошибка: delay должен быть >= 0, timeout должен быть > 0", file=sys.stderr)
        return 2

    try:
        targets = _targeted_config()
        overrides = _overrides_config()
        reasons = {target["source_url"]: target["reason"] for target in targets}
        with Database.from_env() as database:
            products = database.available_products_for_enrichment()
        selected = select_pilot_products(
            products,
            [target["source_url"] for target in targets],
            limit=args.limit,
        )

        with BladeRubberClient(
            delay=args.delay,
            timeout=args.timeout,
            max_reviews=args.max_reviews,
        ) as client:
            print("Загрузка публичного sitemap BladeRubber.net...")
            catalog = client.fetch_catalog()
            print(f"Внешних карточек оснований: {len(catalog)}")
            results: list[dict[str, Any]] = []
            for index, product in enumerate(selected, start=1):
                print(f"Сопоставление {index}/{len(selected)}: {product['name']}")
                match = _match_product(
                    product,
                    catalog,
                    client,
                    overrides.get(product["source_url"]),
                )
                results.append(
                    {
                        "product_id": product["id"],
                        "name": product["name"],
                        "brand": product["brand"],
                        "source_url": product["source_url"],
                        "cohort": product["cohort"],
                        "selection_reason": reasons.get(product["source_url"]),
                        "match": match,
                    }
                )
    except (ConfigurationError, psycopg.Error, requests.RequestException, ValueError) as error:
        print(f"Ошибка enrichment: {redact_secrets(str(error))}", file=sys.stderr)
        return 1

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "name": SOURCE_NAME,
            "robots_url": ROBOTS_URL,
            "terms_url": None,
            "raw_review_text_retained": False,
            "collection_scope": "limited 20-product research pilot",
        },
        "selection": {
            "method": "10 preregistered eval targets plus deterministic brand-stratified coverage sample",
            "requested_limit": args.limit,
        },
        "metrics": _metrics(results, len(catalog)),
        "products": results,
    }
    _write_json(args.output, payload)
    metrics = payload["metrics"]
    print(f"Сопоставлено: {metrics['matched_products']}/{metrics['selected_products']}")
    print(f"Требуют ручной проверки: {metrics['candidate_products']}")
    print(f"Отзывов агрегировано: {metrics['reviews_sampled']}")
    print(f"Результат: {args.output}")
    return 0
