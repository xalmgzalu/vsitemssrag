"""Загрузка локальных настроек приложения."""

from __future__ import annotations

import os
from pathlib import Path


class ConfigurationError(RuntimeError):
    """Ошибка конфигурации приложения."""


def find_env_file() -> Path:
    candidates = (
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def load_env(path: Path | None = None) -> None:
    """Загрузить простые пары KEY=VALUE из .env без перезаписи окружения."""
    env_path = path or find_env_file()
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if key:
            os.environ.setdefault(key, value)


def require_setting(name: str) -> str:
    load_env()
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"добавьте {name} в файл .env")
    return value
