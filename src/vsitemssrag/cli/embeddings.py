"""CLI для построения embeddings и semantic search."""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal

import psycopg
import requests

from vsitemssrag.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    GeminiEmbeddingClient,
    GeminiEmbeddingError,
    build_product_document,
    build_search_query,
)
from vsitemssrag.settings import ConfigurationError, require_setting
from vsitemssrag.storage.postgres import Database, redact_secrets


def positive_or_zero(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("значение не может быть отрицательным")
    return parsed


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return parsed


def configure_embed_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit",
        type=positive_or_zero,
        default=3,
        help="максимум товаров; 0 — все ожидающие (по умолчанию: 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="пересчитать даже актуальные embeddings",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="пауза между Gemini API-запросами (по умолчанию: 1.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="таймаут Gemini API-запроса (по умолчанию: 45)",
    )
    parser.set_defaults(handler=run_embed)


def configure_search_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query", help="поисковый запрос на естественном языке")
    parser.add_argument(
        "--limit",
        type=positive,
        default=5,
        help="число результатов (по умолчанию: 5)",
    )
    parser.add_argument(
        "--include-unavailable",
        action="store_true",
        help="включить недоступные товары",
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.set_defaults(handler=run_search)


def run_embed(args: argparse.Namespace) -> int:
    if args.delay < 0 or args.timeout <= 0:
        print("Ошибка: delay должен быть >= 0, timeout должен быть > 0", file=sys.stderr)
        return 2

    saved = 0
    failed = 0
    try:
        api_key = require_setting("GEMINI_API_KEY")
        with Database.from_env() as database:
            database.apply_migrations()
            products = database.products_for_embedding(
                model=EMBEDDING_MODEL,
                limit=args.limit,
                force=args.force,
            )
            if not products:
                print("Все embeddings уже актуальны.")
                return 0

            print(
                f"К обработке: {len(products)}; модель: {EMBEDDING_MODEL}; "
                f"размерность: {EMBEDDING_DIMENSIONS}"
            )
            with GeminiEmbeddingClient(
                api_key,
                delay=args.delay,
                timeout=args.timeout,
            ) as client:
                for index, product in enumerate(products, start=1):
                    print(f"Товар {index}/{len(products)}: {product['name']}")
                    try:
                        content = build_product_document(product)
                        embedding = client.embed(content)
                        database.upsert_product_embedding(
                            product_id=product["id"],
                            model=EMBEDDING_MODEL,
                            content_hash=product["content_hash"],
                            content=content,
                            embedding=embedding,
                        )
                    except (GeminiEmbeddingError, requests.RequestException, ValueError) as error:
                        print(f"  Пропущен: {error}", file=sys.stderr)
                        failed += 1
                        continue
                    saved += 1
    except (ConfigurationError, psycopg.Error) as error:
        print(f"Ошибка: {redact_secrets(str(error))}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nОстановлено пользователем; готовые embeddings сохранены.")
        return 130

    print(f"Готово: сохранено embeddings — {saved}")
    if failed:
        print(f"С ошибками: {failed}", file=sys.stderr)
        return 1
    return 0


def format_price(price: Decimal | None, currency: str | None) -> str:
    if price is None:
        return "цена не указана"
    formatted = f"{price:,.0f}".replace(",", " ")
    return f"{formatted} {currency or ''}".strip()


def run_search(args: argparse.Namespace) -> int:
    if args.timeout <= 0:
        print("Ошибка: timeout должен быть > 0", file=sys.stderr)
        return 2

    try:
        api_key = require_setting("GEMINI_API_KEY")
        query = build_search_query(args.query)
        with GeminiEmbeddingClient(api_key, delay=0, timeout=args.timeout) as client:
            query_embedding = client.embed(query)

        with Database.from_env() as database:
            database.apply_migrations()
            results = database.match_products(
                query_embedding,
                model=EMBEDDING_MODEL,
                limit=args.limit,
                only_available=not args.include_unavailable,
            )
    except (
        ConfigurationError,
        GeminiEmbeddingError,
        psycopg.Error,
        requests.RequestException,
        ValueError,
    ) as error:
        print(f"Ошибка поиска: {redact_secrets(str(error))}", file=sys.stderr)
        return 1

    if not results:
        print("Ничего не найдено. Сначала запустите команду embed.")
        return 0

    for index, product in enumerate(results, start=1):
        print(
            f"{index}. {product['name']} — {format_price(product['price'], product['currency'])}"
        )
        print(
            f"   бренд: {product['brand']}; similarity: {product['similarity']:.4f}"
        )
        print(f"   {product['source_url']}")
    return 0
