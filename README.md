# VistaSport RAG

Локальный RAG-проект для поиска и подбора инвентаря для настольного тенниса.

## Установка

Установите `uv` на macOS и синхронизируйте окружение:

```bash
brew install uv
uv sync
```

`uv` создаст `.venv`, установит проект в editable-режиме и зафиксированные в
`uv.lock` зависимости.

## Команды

Все операции доступны через единый CLI:

```bash
uv run vsitemssrag --help
```

Проверка подключения и управление схемой PostgreSQL:

```bash
uv run vsitemssrag db check
uv run vsitemssrag db migrate
uv run vsitemssrag db status
```

Безопасный пробный запуск парсера ограничен десятью товарами. По умолчанию JSON
будет записан в `data/exports/products.json`:

```bash
uv run vsitemssrag scrape
```

Потоковая запись трёх товаров напрямую в Supabase PostgreSQL:

```bash
uv run vsitemssrag scrape --storage postgres --limit 3
```

Полный проход без создания JSON:

```bash
uv run vsitemssrag scrape --storage postgres --limit 0
```

Одновременная запись в PostgreSQL и отладочный JSON:

```bash
uv run vsitemssrag scrape --storage both --limit 10
```

Парсер делает паузу 1,5 секунды между запросами, повторяет временно неудачные
запросы и сохраняет каждую карточку сразу после обработки. Ограниченный или
частично неудачный запуск не помечает остальные товары отсутствующими. Только
полностью успешный запуск с `--limit 0` помечает пропавшие товары недоступными.

## Конфигурация

Секреты хранятся в `.env`, который исключён из Git:

```env
GEMINI_API_KEY=
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/postgres
```

## Векторное хранилище

Команда миграции включает расширение `pgvector` в Supabase и создаёт таблицу
`public.product_embeddings`:

```bash
uv run vsitemssrag db migrate
```

Для embeddings выбрана модель `gemini-embedding-2` с размерностью 768. Таблица
хранит один актуальный embedding на товар вместе с моделью, исходным текстом и
`content_hash`. Поиск использует cosine distance через HNSW-индекс и серверную
функцию `public.match_products`. На таблице включён RLS без клиентских политик.

Сразу после миграции таблица пуста. Её заполняет отдельный embedding pipeline,
который будет обрабатывать только товары с новым или изменившимся
`content_hash`.

Создайте ключ на странице [Google AI Studio API Keys](https://aistudio.google.com/app/apikey)
и сохраните его только в локальном `.env`:

```env
GEMINI_API_KEY=your-key
```

Безопасный пробный запуск обрабатывает три товара:

```bash
uv run vsitemssrag embed
uv run vsitemssrag search "контрольное основание для новичка"
```

После проверки качества можно обработать все ожидающие товары:

```bash
uv run vsitemssrag embed --limit 0
```

Цена и наличие не входят в embedding-текст: semantic search отвечает за смысл,
а изменчивые и точные ограничения применяются к актуальным полям PostgreSQL.

## Структура проекта

```text
.
├── data/                         # локальные генерируемые данные
│   └── exports/
├── docs/                         # проектная документация
├── src/vsitemssrag/              # устанавливаемый Python-пакет
│   ├── cli/                      # команды приложения
│   ├── scraping/                 # HTTP и разбор VistaSport
│   ├── storage/                  # PostgreSQL и SQL-миграции
│   ├── models.py                 # общие типы данных
│   └── settings.py               # загрузка настроек
├── tests/unit/                   # быстрые изолированные тесты
├── pyproject.toml                # метаданные и зависимости
└── uv.lock                       # точные версии зависимостей
```

## Тесты и сборка

```bash
uv run python -m unittest discover -s tests -v
uv build
```

## Управление зависимостями

```bash
uv add package-name
uv remove package-name
uv lock --upgrade
```
