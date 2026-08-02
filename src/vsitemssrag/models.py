"""Типы данных каталога, общие для парсера и хранилищ."""

from __future__ import annotations

from typing import TypedDict


Price = int | float | None


class ProductVariant(TypedDict):
    name: str | None
    sku: str | None
    price: Price
    currency: str | None
    available: bool


class Product(TypedDict):
    name: str
    link: str
    price: Price
    currency: str | None
    available: bool
    brand: str | None
    description: str
    characteristics: dict[str, str]
    variants: list[ProductVariant]
    image: str | None
