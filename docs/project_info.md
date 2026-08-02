# Архитектура VistaSport RAG

## Назначение

Проект собирает доступные основания для настольного тенниса из каталога
VistaSport, нормализует карточки и сохраняет их в JSON или Supabase PostgreSQL.
Следующий слой проекта — построение embeddings и RAG-поиска по сохранённым
товарам.

## Слои приложения

### `scraping`

`vsitemssrag.scraping.vistasport` отвечает только за внешний сайт:

- HTTP-сессию, задержку и повтор временных ошибок;
- обход пагинации каталога;
- извлечение ссылок на карточки;
- разбор ProductGroup JSON-LD и полного HTML-описания;
- преобразование результата в общий тип `Product`.

Модуль не знает о CLI, `.env` или конкретном хранилище.

### `storage`

`vsitemssrag.storage.postgres` отвечает за PostgreSQL:

- применение версионированных SQL-миграций;
- идемпотентный UPSERT товаров и вариантов;
- учёт запусков парсера;
- пометку пропавших товаров недоступными после полного успешного обхода;
- вычисление `content_hash` для будущего обновления embeddings.

SQL-файлы находятся в `vsitemssrag.storage.migrations` и входят в пакет.

### `cli`

`vsitemssrag.cli` связывает парсер и хранилища, обрабатывает аргументы командной
строки и выводит прогресс. Единственная пользовательская точка входа:

```bash
uv run vsitemssrag <command>
```

### Конфигурация и модели

- `vsitemssrag.settings` загружает секреты из корневого `.env`.
- `vsitemssrag.models` содержит общие типы `Product` и `ProductVariant`.

## Поток данных

```text
VistaSport catalog
        │
        ▼
scraping.vistasport
        │ Product
        ├──────────────► data/exports/*.json
        │
        ▼
storage.postgres ──────► Supabase PostgreSQL
                              │
                              ▼
                    embeddings / RAG (следующий этап)
```

## Таблицы

- `products` — нормализованная карточка и характеристики JSONB.
- `product_variants` — варианты ручек, SKU, цены и наличие.
- `scrape_runs` — состояние и метрики каждого запуска.
- `app_private.schema_migrations` — применённые версии схемы.

На таблицах публичной схемы включён RLS без клиентских политик. Текущая запись
выполняется только серверным PostgreSQL-подключением из `DATABASE_URL`.
