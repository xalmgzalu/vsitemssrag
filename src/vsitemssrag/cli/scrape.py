"""CLI-оркестрация парсинга и сохранения каталога."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import psycopg
import requests

from vsitemssrag.models import Product
from vsitemssrag.scraping.vistasport import (
    DEFAULT_CATALOG_URL,
    HttpClient,
    collect_product_urls,
    parse_product_page,
)
from vsitemssrag.settings import ConfigurationError
from vsitemssrag.storage.postgres import Database, redact_secrets


DEFAULT_EXPORT_PATH = Path("data/exports/products.json")


def positive_or_zero(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("значение не может быть отрицательным")
    return parsed


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog-url", default=DEFAULT_CATALOG_URL)
    parser.add_argument(
        "--storage",
        choices=("json", "postgres", "both"),
        default="json",
        help="хранилище результата (по умолчанию: json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPORT_PATH,
        help=f"путь к JSON (по умолчанию: {DEFAULT_EXPORT_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=positive_or_zero,
        default=10,
        help="максимум товаров; 0 — весь каталог (по умолчанию: 10)",
    )
    parser.add_argument(
        "--max-pages",
        type=positive_or_zero,
        default=0,
        help="максимум страниц каталога; 0 — без ограничения",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="пауза между HTTP-запросами в секундах (по умолчанию: 1.5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="таймаут чтения одного запроса в секундах (по умолчанию: 45)",
    )
    parser.set_defaults(handler=run)


def save_products(products: list[Product], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(products, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def finish_run_safely(
    database: Database | None,
    run_id: uuid.UUID | None,
    **values: Any,
) -> None:
    if not database or not run_id:
        return
    try:
        database.finish_run(run_id, **values)
    except psycopg.Error:
        pass


def run(args: argparse.Namespace) -> int:
    if args.delay < 0 or args.timeout <= 0:
        print("Ошибка: delay должен быть >= 0, timeout должен быть > 0", file=sys.stderr)
        return 2

    products: list[Product] = []
    database: Database | None = None
    run_id: uuid.UUID | None = None
    run_finished = False
    saved_count = 0
    failed_count = 0
    write_json = args.storage in {"json", "both"}
    write_postgres = args.storage in {"postgres", "both"}
    full_scan_requested = args.limit == 0 and args.max_pages == 0

    with HttpClient(delay=args.delay, timeout=args.timeout) as client:
        try:
            if write_postgres:
                database = Database.from_env()
                applied_migrations = database.apply_migrations()
                if applied_migrations:
                    print("Применены миграции: " + ", ".join(applied_migrations))
                run_id = database.start_run(
                    catalog_url=args.catalog_url,
                    storage_mode=args.storage,
                    full_scan=full_scan_requested,
                )

            product_urls = collect_product_urls(
                client=client,
                catalog_url=args.catalog_url,
                limit=args.limit,
                max_pages=args.max_pages,
                on_page=lambda number, _url: print(f"Каталог: страница {number}"),
            )
            print(f"Найдено товаров для загрузки: {len(product_urls)}")
            if database and run_id:
                database.set_products_found(run_id, len(product_urls))

            if not product_urls:
                finish_run_safely(
                    database,
                    run_id,
                    status="failed",
                    products_saved=0,
                    products_failed=0,
                    error_message="каталог не вернул товары",
                )
                run_finished = True
                print("Каталог не вернул ни одного товара.", file=sys.stderr)
                return 1

            for index, product_url in enumerate(product_urls, start=1):
                print(f"Товар {index}/{len(product_urls)}: {product_url}")
                try:
                    product = parse_product_page(
                        client.get_text(product_url), product_url
                    )
                except (requests.RequestException, ValueError) as error:
                    print(f"  Пропущен: {error}", file=sys.stderr)
                    failed_count += 1
                    continue

                try:
                    if database and run_id:
                        database.upsert_product(product, run_id)
                    if write_json:
                        products.append(product)
                        save_products(products, args.output)
                except (psycopg.Error, OSError, ValueError) as error:
                    print(
                        f"  Не удалось сохранить: {redact_secrets(str(error))}",
                        file=sys.stderr,
                    )
                    failed_count += 1
                    continue

                saved_count += 1

            status = "completed" if failed_count == 0 else "partial"
            complete_full_scan = (
                full_scan_requested
                and failed_count == 0
                and saved_count == len(product_urls)
            )
            if database and run_id:
                database.finish_run(
                    run_id,
                    status=status,
                    products_saved=saved_count,
                    products_failed=failed_count,
                    deactivate_missing=complete_full_scan,
                )
                run_finished = True
        except requests.RequestException as error:
            finish_run_safely(
                database,
                run_id,
                status="failed",
                products_saved=saved_count,
                products_failed=failed_count,
                error_message=str(error),
            )
            run_finished = True
            print(f"Ошибка загрузки каталога: {error}", file=sys.stderr)
            return 1
        except (ConfigurationError, psycopg.Error) as error:
            finish_run_safely(
                database,
                run_id,
                status="failed",
                products_saved=saved_count,
                products_failed=failed_count,
                error_message=redact_secrets(str(error)),
            )
            run_finished = True
            print(
                f"Ошибка PostgreSQL: {redact_secrets(str(error))}",
                file=sys.stderr,
            )
            return 1
        except KeyboardInterrupt:
            finish_run_safely(
                database,
                run_id,
                status="cancelled",
                products_saved=saved_count,
                products_failed=failed_count,
            )
            run_finished = True
            print("\nОстановлено пользователем; обработанные товары сохранены.")
            return 130
        finally:
            if database:
                if run_id and not run_finished:
                    finish_run_safely(
                        database,
                        run_id,
                        status="failed",
                        products_saved=saved_count,
                        products_failed=failed_count,
                        error_message="неожиданное завершение запуска",
                    )
                database.close()

    if saved_count == 0:
        print("Не удалось получить ни одного товара.", file=sys.stderr)
        return 1

    destinations: list[str] = []
    if write_json:
        destinations.append(str(args.output))
    if write_postgres:
        destinations.append("PostgreSQL")
    print(f"Готово: {saved_count} товаров сохранено в {', '.join(destinations)}")
    if failed_count:
        print(f"С ошибками: {failed_count}", file=sys.stderr)
    return 0
