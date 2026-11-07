"""CLI настоящего RAG-ответа поверх semantic search."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg
import requests

from vsitemssrag.cli.embeddings import format_price, positive
from vsitemssrag.embeddings import (
    EMBEDDING_MODEL,
    GeminiEmbeddingClient,
    GeminiEmbeddingError,
    build_search_query,
)
from vsitemssrag.rag import (
    CLARIFYING_QUESTION,
    GENERATION_MODEL,
    GeminiGenerationClient,
    GeminiGenerationError,
    extract_max_price_rub,
    generate_advisor_answer,
    no_budget_matches_answer,
    requires_clarification,
    select_context_products,
)
from vsitemssrag.settings import ConfigurationError, load_env, require_setting
from vsitemssrag.storage.postgres import Database, redact_secrets


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("question", help="вопрос консультанту на естественном языке")
    parser.add_argument(
        "--limit",
        type=positive,
        default=6,
        help="число товаров в RAG-контексте (по умолчанию: 6)",
    )
    parser.add_argument(
        "--include-unavailable",
        action="store_true",
        help="разрешить рекомендации недоступных товаров",
    )
    parser.add_argument(
        "--model",
        help="Gemini generation model; иначе GEMINI_GENERATION_MODEL или default",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=positive,
        default=1200,
        help="максимальная длина ответа Gemini (по умолчанию: 1200)",
    )
    parser.add_argument(
        "--evidence-file",
        type=Path,
        help="JSON пилота внешних отзывов; используются только matched-сопоставления",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.set_defaults(handler=run_ask)


def _generation_model(cli_model: str | None) -> str:
    load_env()
    return cli_model or os.getenv("GEMINI_GENERATION_MODEL", "").strip() or GENERATION_MODEL


def load_community_evidence(path: Path | None) -> dict[int, dict[str, Any]]:
    """Загрузить только автоматически принятые или вручную проверенные match."""
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_products = payload.get("products") if isinstance(payload, Mapping) else None
    if not isinstance(raw_products, list):
        raise ValueError("evidence-файл не содержит список products")

    evidence_by_id: dict[int, dict[str, Any]] = {}
    for product in raw_products:
        if not isinstance(product, Mapping):
            continue
        match = product.get("match")
        if not isinstance(match, Mapping) or match.get("status") != "matched":
            continue
        evidence = match.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        try:
            product_id = int(product["product_id"])
        except (KeyError, TypeError, ValueError):
            continue
        evidence_by_id[product_id] = dict(evidence)
    return evidence_by_id


def attach_community_evidence(
    products: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for product in products:
        item = dict(product)
        try:
            evidence = evidence_by_id.get(int(product.get("product_id")))
        except (TypeError, ValueError):
            evidence = None
        if evidence is not None:
            item["community_evidence"] = dict(evidence)
        enriched.append(item)
    return enriched


def run_ask(args: argparse.Namespace) -> int:
    if args.timeout <= 0:
        print("Ошибка: timeout должен быть > 0", file=sys.stderr)
        return 2
    if requires_clarification(args.question):
        print(CLARIFYING_QUESTION)
        return 0

    try:
        api_key = require_setting("GEMINI_API_KEY")
        model = _generation_model(args.model)
        evidence_by_id = load_community_evidence(args.evidence_file)
        search_query = build_search_query(args.question)
        max_price = extract_max_price_rub(args.question)

        with GeminiEmbeddingClient(api_key, delay=0, timeout=args.timeout) as client:
            query_embedding = client.embed(search_query)

        with Database.from_env() as database:
            database.apply_migrations()
            candidate_limit = (
                min(max(args.limit * 5, 20), 100) if max_price else args.limit
            )
            candidates = database.match_products(
                query_embedding,
                model=EMBEDDING_MODEL,
                limit=candidate_limit,
                only_available=not args.include_unavailable,
            )

        products = select_context_products(candidates, max_price, limit=args.limit)

        if not products:
            if max_price is not None:
                print(no_budget_matches_answer(max_price))
                return 0
            print("Подходящие товары не найдены. Сначала запустите команду embed.")
            return 0

        products = attach_community_evidence(products, evidence_by_id)

        with GeminiGenerationClient(
            api_key,
            model=model,
            timeout=args.timeout,
        ) as client:
            answer = generate_advisor_answer(
                client,
                args.question,
                products,
                max_output_tokens=args.max_output_tokens,
            )
    except (
        ConfigurationError,
        GeminiEmbeddingError,
        GeminiGenerationError,
        psycopg.Error,
        requests.RequestException,
        ValueError,
    ) as error:
        print(f"Ошибка RAG: {redact_secrets(str(error))}", file=sys.stderr)
        return 1

    print(answer.strip())
    print("\nИсточники каталога:")
    for index, product in enumerate(products, start=1):
        price = format_price(product.get("price"), product.get("currency"))
        print(f"[{index}] {product['name']} — {price}")
        print(f"    {product['source_url']}")
        evidence = product.get("community_evidence")
        if isinstance(evidence, Mapping) and evidence.get("source_url"):
            print(f"    Отзывы: {evidence['source_url']}")
    return 0
