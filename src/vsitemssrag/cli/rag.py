"""CLI RAG-консультанта поверх общего сервиса подбора."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

from vsitemssrag.cli.embeddings import format_price, positive
from vsitemssrag.rag.evidence import attach_community_evidence, load_community_evidence
from vsitemssrag.rag.service import RAG_ERRORS, ask_catalog
from vsitemssrag.storage.postgres import redact_secrets

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



def run_ask(args: argparse.Namespace) -> int:
    if args.timeout <= 0:
        print("Ошибка: timeout должен быть > 0", file=sys.stderr)
        return 2
    try:
        result = ask_catalog(
            args.question, limit=args.limit,
            include_unavailable=args.include_unavailable, model=args.model,
            max_output_tokens=args.max_output_tokens,
            evidence_file=args.evidence_file, timeout=args.timeout,
        )
    except RAG_ERRORS as error:
        print(f"Ошибка RAG: {redact_secrets(str(error))}", file=sys.stderr)
        return 1

    print(result.text.strip())
    if result.products:
        print("\nИсточники каталога:")
    for index, product in enumerate(result.products, start=1):
        price = format_price(product.get("price"), product.get("currency"))
        print(f"[{index}] {product['name']} — {price}")
        print(f"    {product['source_url']}")
        evidence = product.get("community_evidence")
        if isinstance(evidence, Mapping) and evidence.get("source_url"):
            print(f"    Отзывы: {evidence['source_url']}")
    return 0
