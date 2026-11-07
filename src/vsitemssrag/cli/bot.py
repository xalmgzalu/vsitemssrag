"""CLI Telegram-бота; внешние клиенты импортируются только при запуске."""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import sqlite3
import sys
from pathlib import Path

from vsitemssrag.cli.embeddings import positive
from vsitemssrag.rag.evidence import load_community_evidence
from vsitemssrag.settings import ConfigurationError, load_env, require_setting


def configure_parser(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="bot_command", required=True)
    run = actions.add_parser("run", help="запустить Telegram long-polling")
    run.add_argument("--evidence-file", type=Path, help="агрегаты отзывов; иначе RAG_EVIDENCE_FILE или пилот v2")
    run.add_argument("--feedback-db", type=Path, default=Path("data/bot/feedback.sqlite3"))
    run.add_argument("--timeout", type=float, default=30.0, help="timeout RAG-запросов, секунд")
    run.add_argument("--max-concurrent", type=positive, default=2)
    run.add_argument("--check", action="store_true", help="проверить конфигурацию и Telegram без polling")
    run.set_defaults(handler=run_bot)


class SecretRedactingFormatter(logging.Formatter):
    def __init__(self, secrets: list[str]) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self.secrets = [value for value in secrets if value]

    def format(self, record: logging.LogRecord) -> str:
        output = super().format(record)
        for value in self.secrets:
            output = output.replace(value, "[REDACTED]")
        return output


def run_bot(args: argparse.Namespace) -> int:
    from aiogram.exceptions import TelegramAPIError, TelegramUnauthorizedError
    from aiogram.utils.token import TokenValidationError
    from vsitemssrag.bot.bot import BotConfig, run_polling

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        print("Ошибка: timeout должен быть положительным конечным числом", file=sys.stderr)
        return 2
    try:
        load_env()
        token = require_setting("TELEGRAM_BOT_TOKEN")
        require_setting("GEMINI_API_KEY")
        require_setting("DATABASE_URL")
        evidence = args.evidence_file or Path(
            os.getenv("RAG_EVIDENCE_FILE", "").strip()
            or "data/exports/review_enrichment_pilot_v2.json"
        )
        if not load_community_evidence(evidence):
            raise ConfigurationError("для бота нужен evidence-файл с matched-сопоставлениями")
        config = BotConfig(evidence, args.feedback_db, args.timeout, args.max_concurrent)
        handler = logging.StreamHandler()
        handler.setFormatter(SecretRedactingFormatter([
            token, os.getenv("GEMINI_API_KEY", ""), os.getenv("DATABASE_URL", ""),
        ]))
        logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
        asyncio.run(run_polling(token, config, check_only=args.check))
    except KeyboardInterrupt:
        print("Бот остановлен.")
    except TokenValidationError:
        print("Ошибка: TELEGRAM_BOT_TOKEN имеет неверный формат.", file=sys.stderr)
        return 1
    except TelegramUnauthorizedError:
        print("Ошибка: Telegram отклонил токен бота.", file=sys.stderr)
        return 1
    except TelegramAPIError as error:
        print(f"Ошибка связи с Telegram ({type(error).__name__}). Проверьте соединение и настройки бота.", file=sys.stderr)
        return 1
    except (ConfigurationError, ValueError, OSError, RuntimeError, sqlite3.Error) as error:
        # Тексты исключений сторонних библиотек могут содержать секреты.
        message = str(error)
        for key in ("TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY", "DATABASE_URL"):
            value = os.getenv(key, "")
            if value:
                message = message.replace(value, "[REDACTED]")
        print(f"Ошибка запуска бота: {message}", file=sys.stderr)
        return 1
    return 0
