"""Команды управления PostgreSQL."""

from __future__ import annotations

import argparse
import sys

import psycopg

from vsitemssrag.settings import ConfigurationError
from vsitemssrag.storage.postgres import Database, redact_secrets


def configure_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="database_command", required=True)
    commands.add_parser("migrate", help="применить SQL-миграции")
    commands.add_parser("status", help="показать статистику каталога")
    commands.add_parser("check", help="проверить подключение к PostgreSQL")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    try:
        with Database.from_env() as database:
            if args.database_command == "check":
                return check_connection(database)

            if args.database_command == "migrate":
                applied = database.apply_migrations()
                if applied:
                    print("Применены миграции: " + ", ".join(applied))
                else:
                    print("Схема базы уже актуальна.")
                return 0

            database.apply_migrations()
            products, available, variants = database.stats()
            print(f"Товаров: {products}")
            print(f"Доступных товаров: {available}")
            print(f"Вариантов: {variants}")
            return 0
    except (ConfigurationError, psycopg.Error) as error:
        print(f"Ошибка базы данных: {redact_secrets(str(error))}", file=sys.stderr)
        return 1


def check_connection(database: Database) -> int:
    database_name, user, server_time, public_tables = database.connection.execute(
        """
        select
            current_database(),
            current_user,
            current_timestamp,
            (
                select count(*)
                from information_schema.tables
                where table_schema = 'public'
            )
        """
    ).fetchone()

    print("Подключение к PostgreSQL успешно.")
    print(f"База: {database_name}")
    print(f"Пользователь: {user}")
    print(f"Время сервера: {server_time}")
    print(f"Таблиц в схеме public: {public_tables}")
    return 0
