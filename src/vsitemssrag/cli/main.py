"""Единая точка входа для команд проекта."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from vsitemssrag.cli import database as database_cli
from vsitemssrag.cli import embeddings as embeddings_cli
from vsitemssrag.cli import rag as rag_cli
from vsitemssrag.cli import scrape as scrape_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vsitemssrag",
        description="VistaSport RAG utilities",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    scrape_parser = commands.add_parser("scrape", help="запустить парсер каталога")
    scrape_cli.configure_parser(scrape_parser)

    database_parser = commands.add_parser("db", help="управление PostgreSQL")
    database_cli.configure_parser(database_parser)

    embed_parser = commands.add_parser(
        "embed", help="построить embeddings товаров"
    )
    embeddings_cli.configure_embed_parser(embed_parser)

    search_parser = commands.add_parser(
        "search", help="семантический поиск по каталогу"
    )
    embeddings_cli.configure_search_parser(search_parser)

    ask_parser = commands.add_parser(
        "ask", help="получить grounded-рекомендацию по каталогу"
    )
    rag_cli.configure_parser(ask_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
