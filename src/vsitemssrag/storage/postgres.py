"""PostgreSQL-хранилище каталога VistaSport."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from vsitemssrag.models import Product, ProductVariant
from vsitemssrag.settings import require_setting


MIGRATIONS_DIR = Path(__file__).with_name("migrations")


def redact_secrets(message: str) -> str:
    message = re.sub(
        r"(postgres(?:ql)?://[^:/?#\s]+:)[^@/\s]+@",
        r"\1***@",
        message,
        flags=re.IGNORECASE,
    )
    return re.sub(r"(password\s*=\s*)\S+", r"\1***", message, flags=re.IGNORECASE)


def product_content_hash(product: Mapping[str, Any]) -> str:
    """Хеш полей, которые позднее будут использоваться в RAG-контексте."""
    content = {
        "name": product.get("name"),
        "brand": product.get("brand"),
        "description": product.get("description"),
        "price": product.get("price"),
        "currency": product.get("currency"),
        "characteristics": product.get("characteristics") or {},
    }
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def variant_source_key(variant: Mapping[str, Any]) -> str:
    sku = str(variant.get("sku") or "").strip()
    if sku:
        return f"sku:{sku}"

    name = str(variant.get("name") or "").strip()
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return f"name:{digest}"


class Database:
    def __init__(self, database_url: str) -> None:
        self.connection = psycopg.connect(
            database_url,
            sslmode="require",
            connect_timeout=10,
            autocommit=True,
            prepare_threshold=None,
        )

    @classmethod
    def from_env(cls) -> "Database":
        return cls(require_setting("DATABASE_URL"))

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def apply_migrations(self, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
        self.connection.execute("create schema if not exists app_private")
        self.connection.execute(
            """
            create table if not exists app_private.schema_migrations (
                version text primary key,
                applied_at timestamptz not null default now()
            )
            """
        )
        applied = {
            row[0]
            for row in self.connection.execute(
                "select version from app_private.schema_migrations"
            ).fetchall()
        }

        newly_applied: list[str] = []
        for migration_path in sorted(migrations_dir.glob("*.sql")):
            version = migration_path.name
            if version in applied:
                continue

            sql = migration_path.read_text(encoding="utf-8")
            with self.connection.transaction():
                self.connection.execute(sql)
                self.connection.execute(
                    "insert into app_private.schema_migrations (version) values (%s)",
                    (version,),
                )
            newly_applied.append(version)

        return newly_applied

    def start_run(
        self,
        *,
        catalog_url: str,
        storage_mode: str,
        full_scan: bool,
    ) -> uuid.UUID:
        run_id = uuid.uuid4()
        self.connection.execute(
            """
            insert into public.scrape_runs (
                id, source, category, catalog_url, storage_mode, full_scan
            ) values (%s, 'vistasport', 'blades', %s, %s, %s)
            """,
            (run_id, catalog_url, storage_mode, full_scan),
        )
        return run_id

    def set_products_found(self, run_id: uuid.UUID, count: int) -> None:
        self.connection.execute(
            "update public.scrape_runs set products_found = %s where id = %s",
            (count, run_id),
        )

    def upsert_product(self, product: Product, run_id: uuid.UUID) -> int:
        source_url = product["link"].strip()
        name = product["name"].strip()
        if not source_url or not name:
            raise ValueError("у товара отсутствует link или name")

        with self.connection.transaction():
            product_id = self.connection.execute(
                """
                insert into public.products (
                    source, category, source_url, name, brand, description,
                    price, currency, available, characteristics, image_url,
                    content_hash, last_seen_run_id
                ) values (
                    'vistasport', 'blades', %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                )
                on conflict (source_url) do update set
                    name = excluded.name,
                    brand = excluded.brand,
                    description = excluded.description,
                    price = excluded.price,
                    currency = excluded.currency,
                    available = excluded.available,
                    characteristics = excluded.characteristics,
                    image_url = excluded.image_url,
                    content_hash = excluded.content_hash,
                    last_seen_run_id = excluded.last_seen_run_id,
                    last_seen_at = now(),
                    updated_at = now()
                returning id
                """,
                (
                    source_url,
                    name,
                    product["brand"],
                    product["description"],
                    product["price"],
                    product["currency"],
                    product["available"],
                    Jsonb(product["characteristics"]),
                    product["image"],
                    product_content_hash(product),
                    run_id,
                ),
            ).fetchone()[0]

            active_variant_keys = self._upsert_variants(
                product_id=product_id,
                product_name=name,
                variants=product["variants"],
            )
            self._deactivate_missing_variants(product_id, active_variant_keys)

        return product_id

    def _upsert_variants(
        self,
        *,
        product_id: int,
        product_name: str,
        variants: list[ProductVariant],
    ) -> list[str]:
        active_variant_keys: list[str] = []
        for variant in variants:
            source_key = variant_source_key(variant)
            active_variant_keys.append(source_key)
            self.connection.execute(
                """
                insert into public.product_variants (
                    product_id, source_key, sku, name, price, currency, available
                ) values (%s, %s, %s, %s, %s, %s, %s)
                on conflict (product_id, source_key) do update set
                    sku = excluded.sku,
                    name = excluded.name,
                    price = excluded.price,
                    currency = excluded.currency,
                    available = excluded.available,
                    updated_at = now()
                """,
                (
                    product_id,
                    source_key,
                    variant["sku"],
                    variant["name"] or product_name,
                    variant["price"],
                    variant["currency"],
                    variant["available"],
                ),
            )
        return active_variant_keys

    def _deactivate_missing_variants(
        self, product_id: int, active_variant_keys: list[str]
    ) -> None:
        if active_variant_keys:
            self.connection.execute(
                """
                update public.product_variants
                set available = false, updated_at = now()
                where product_id = %s
                  and not (source_key = any(%s::text[]))
                """,
                (product_id, active_variant_keys),
            )
            return

        self.connection.execute(
            """
            update public.product_variants
            set available = false, updated_at = now()
            where product_id = %s
            """,
            (product_id,),
        )

    def finish_run(
        self,
        run_id: uuid.UUID,
        *,
        status: str,
        products_saved: int,
        products_failed: int,
        error_message: str | None = None,
        deactivate_missing: bool = False,
    ) -> None:
        with self.connection.transaction():
            if deactivate_missing:
                self._deactivate_missing_products(run_id)

            self.connection.execute(
                """
                update public.scrape_runs
                set status = %s,
                    products_saved = %s,
                    products_failed = %s,
                    error_message = %s,
                    finished_at = now()
                where id = %s
                """,
                (
                    status,
                    products_saved,
                    products_failed,
                    error_message,
                    run_id,
                ),
            )

    def _deactivate_missing_products(self, run_id: uuid.UUID) -> None:
        self.connection.execute(
            """
            update public.product_variants as variants
            set available = false, updated_at = now()
            from public.products as products
            where variants.product_id = products.id
              and products.source = 'vistasport'
              and products.category = 'blades'
              and products.last_seen_run_id is distinct from %s
            """,
            (run_id,),
        )
        self.connection.execute(
            """
            update public.products
            set available = false, updated_at = now()
            where source = 'vistasport'
              and category = 'blades'
              and last_seen_run_id is distinct from %s
            """,
            (run_id,),
        )

    def stats(self) -> tuple[int, int, int]:
        products = self.connection.execute(
            "select count(*) from public.products"
        ).fetchone()[0]
        available = self.connection.execute(
            "select count(*) from public.products where available"
        ).fetchone()[0]
        variants = self.connection.execute(
            "select count(*) from public.product_variants"
        ).fetchone()[0]
        return products, available, variants
