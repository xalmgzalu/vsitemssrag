# Локальные генерируемые данные

`exports/` содержит JSON-каталог, агрегаты пилота отзывов, checkpoint A/B,
локальные решения аудита и JSONL-диагностику отклонённых ответов.
Выгрузки и логи исключены из Git; секреты хранятся отдельно в `.env`.
Полные тексты внешних отзывов не сохраняются.

Основные файлы текущей проверки:

- `review_enrichment_pilot_v2.json` — фиксированный evidence-набор.
- `enrichment_ab_results_v5.json` и `enrichment_ab_results_v5_audit.json` —
  завершённый v5 и построчные решения агента.
- `enrichment_ab_results_v6.json` и `enrichment_ab_results_v6_audit.json` —
  завершённый прогон после исправлений по v5 и построчные решения агента.
- `rag_validation_rejections.jsonl` — исходный/итоговый текст и ошибки validation.

Новый клон не содержит этих файлов. Сводки и хеши для передачи проекта — в
[CURRENT_STATE.md](../docs/CURRENT_STATE.md) и
[AB_V5_AUDIT.md](../docs/AB_V5_AUDIT.md) /
[AB_V6_AUDIT.md](../docs/AB_V6_AUDIT.md); команды — в
[evals/README.md](../evals/README.md).


`bot/feedback.sqlite3` содержит локальную историю доставленных ответов и оценки
Telegram-пилота; вся папка `bot/` исключена из Git. Схема и правила подсчёта —
в [TELEGRAM_BOT.md](../docs/TELEGRAM_BOT.md).
