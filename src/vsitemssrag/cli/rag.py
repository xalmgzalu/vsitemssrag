"""CLI настоящего RAG-ответа поверх semantic search."""

from __future__ import annotations

import argparse
import os
import sys

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
    GENERATION_MODEL,
    RAG_SYSTEM_INSTRUCTION,
    GeminiGenerationClient,
    GeminiGenerationError,
    build_rag_prompt,
    extract_max_price_rub,
    products_within_budget,
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
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.set_defaults(handler=run_ask)


def _generation_model(cli_model: str | None) -> str:
    load_env()
    return cli_model or os.getenv("GEMINI_GENERATION_MODEL", "").strip() or GENERATION_MODEL


def run_ask(args: argparse.Namespace) -> int:
    if args.timeout <= 0:
        print("Ошибка: timeout должен быть > 0", file=sys.stderr)
        return 2

    try:
        api_key = require_setting("GEMINI_API_KEY")
        model = _generation_model(args.model)
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

        products = list(candidates[: args.limit])
        if max_price is not None:
            within_budget = products_within_budget(
                candidates,
                max_price,
                limit=args.limit,
            )
            if within_budget:
                products = list(within_budget)

        if not products:
            print("Подходящие товары не найдены. Сначала запустите команду embed.")
            return 0

        prompt = build_rag_prompt(args.question, products)
        with GeminiGenerationClient(
            api_key,
            model=model,
            timeout=args.timeout,
        ) as client:
            answer = client.generate(
                prompt,
                system_instruction=RAG_SYSTEM_INSTRUCTION,
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
    return 0
