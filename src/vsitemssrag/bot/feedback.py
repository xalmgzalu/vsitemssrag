"""Локальная история ответов и одна актуальная оценка на ответ."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

from vsitemssrag.rag.service import AdviceResult


class FeedbackStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("pragma journal_mode=WAL")
            db.executescript("""
                create table if not exists responses (
                    id text primary key,
                    chat_id integer not null,
                    user_id integer not null,
                    message_id integer,
                    question text not null,
                    answer text not null,
                    origin text not null,
                    products_json text not null,
                    created_at text not null
                );
                create table if not exists feedback (
                    response_id text primary key references responses(id),
                    vote integer not null check (vote in (-1, 1)),
                    updated_at text not null
                );
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("pragma foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def create(self, chat_id: int, user_id: int, question: str, result: AdviceResult) -> str:
        response_id = uuid4().hex
        products = [
            {"ref": ref, "product_id": item.get("product_id"), "url": item.get("source_url")}
            for ref, item in result.cited_products
        ]
        with self.connect() as db:
            db.execute(
                "insert into responses values (?, ?, ?, null, ?, ?, ?, ?, ?)",
                (response_id, chat_id, user_id, question, result.text, result.origin,
                 json.dumps(products, ensure_ascii=False), datetime.now(UTC).isoformat()),
            )
        return response_id

    def bind_message(self, response_id: str, message_id: int) -> None:
        with self.connect() as db:
            db.execute("update responses set message_id=? where id=?", (message_id, response_id))

    def vote(self, response_id: str, chat_id: int, user_id: int, message_id: int, vote: int) -> str:
        if vote not in {-1, 1}:
            return "unavailable"
        with self.connect() as db:
            db.execute("begin immediate")
            row = db.execute(
                "select chat_id, user_id, message_id from responses where id=?", (response_id,),
            ).fetchone()
            if row is None or row != (chat_id, user_id, message_id):
                return "unavailable"
            previous = db.execute("select vote from feedback where response_id=?", (response_id,)).fetchone()
            if previous == (vote,):
                return "duplicate"
            db.execute(
                "insert into feedback values (?, ?, ?) on conflict(response_id) do update "
                "set vote=excluded.vote, updated_at=excluded.updated_at",
                (response_id, vote, datetime.now(UTC).isoformat()),
            )
        return "saved"
