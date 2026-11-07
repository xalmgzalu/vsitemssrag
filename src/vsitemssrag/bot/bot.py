"""aiogram long-polling и обработчики первого Telegram-пилота."""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, LinkPreviewOptions, Message
from aiogram.utils.chat_action import ChatActionSender

from vsitemssrag.bot.feedback import FeedbackStore
from vsitemssrag.bot.presentation import answer_keyboard, split_text, telegram_html
from vsitemssrag.rag.service import AdviceResult, RAG_ERRORS, ask_catalog

logger = logging.getLogger(__name__)
WELCOME = (
    "Салам! Я помогу выбрать основание для настольного тенниса из каталога VistaSport.\n\n"
    "Напишите уровень, стиль игры, бюджет и пожелания. Например:\n"
    "«Я новичок, бюджет до 6000 ₽, хочу контрольное основание».\n\n"
    "Отзывы игроков отделяю от характеристик каталога. Каждый запрос рассматриваю "
    "отдельно: после уточнения отправьте полный запрос со всеми критериями."
)


@dataclass(frozen=True)
class BotConfig:
    evidence_file: Path
    feedback_db: Path = Path("data/bot/feedback.sqlite3")
    timeout: float = 30.0
    max_concurrent: int = 2


async def send_text(message: Message, text: str, *, keyboard=None) -> Message:
    """Только повтор явно отклонённого rate-limit запроса; HTML имеет plain fallback."""
    for attempt in range(2):
        try:
            return await message.answer(
                telegram_html(text), parse_mode="HTML", reply_markup=keyboard,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
        except TelegramRetryAfter as error:
            if attempt or error.retry_after > 30:
                raise
            await asyncio.sleep(error.retry_after)
        except TelegramBadRequest as error:
            if not any(term in error.message.casefold() for term in ("parse entit", "unsupported start tag", "find end")):
                raise
            return await message.answer(
                text, parse_mode=None, reply_markup=keyboard,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
    raise RuntimeError("send retry exhausted")


def create_router(
    config: BotConfig, store: FeedbackStore,
    advisor: Callable[..., AdviceResult] = ask_catalog,
) -> Router:
    router = Router(name="advisor")
    active: set[tuple[int, int]] = set()

    async def welcome(message: Message) -> None:
        await send_text(message, WELCOME)

    async def unknown_command(message: Message) -> None:
        await send_text(message, "Доступны /start и /help. Для подбора отправьте вопрос текстом.")

    async def consult(message: Message) -> None:
        if not message.from_user or not message.text:
            return
        key = (message.chat.id, message.from_user.id)
        if key in active:
            await send_text(message, "Предыдущий подбор ещё выполняется. Дождитесь ответа.")
            return
        if len(active) >= config.max_concurrent:
            await send_text(message, "Сейчас выполняются другие подборы. Попробуйте через минуту.")
            return
        active.add(key)
        try:
            async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
                result = await asyncio.to_thread(
                    advisor, message.text, evidence_file=config.evidence_file,
                    timeout=config.timeout,
                )
            response_id = await asyncio.to_thread(
                store.create, message.chat.id, message.from_user.id, message.text, result,
            )
            keyboard = answer_keyboard(result, response_id)
            chunks = split_text(result.text)
            for index, chunk in enumerate(chunks):
                sent = await send_text(
                    message, chunk,
                    keyboard=keyboard if index == len(chunks) - 1 else None,
                )
            await asyncio.to_thread(store.bind_message, response_id, sent.message_id)
        except RAG_ERRORS as error:
            logger.warning("RAG request failed: %s", type(error).__name__)
            await send_text(message, "Не удалось завершить подбор. Попробуйте позже или уточните запрос.")
        except TelegramAPIError as error:
            logger.warning("Telegram delivery failed: %s", type(error).__name__)
        except Exception as error:
            logger.error("Consultation failed: %s", type(error).__name__)
            await send_text(message, "Не удалось обработать запрос. Попробуйте позже.")
        finally:
            active.discard(key)

    async def feedback(callback: CallbackQuery) -> None:
        match = re.fullmatch(r"fb:([a-f0-9]{32}):(up|down)", callback.data or "")
        if not match or not isinstance(callback.message, Message):
            await callback.answer("Эта оценка недоступна.")
            return
        try:
            outcome = await asyncio.to_thread(
                store.vote, match[1], callback.message.chat.id, callback.from_user.id,
                callback.message.message_id, 1 if match[2] == "up" else -1,
            )
        except Exception as error:
            logger.error("Feedback storage failed: %s", type(error).__name__)
            await callback.answer("Не удалось сохранить оценку. Попробуйте позже.")
            return
        replies = {
            "saved": "Спасибо! Оценка сохранена.",
            "duplicate": "Эта оценка уже учтена.",
            "unavailable": "Оценить ответ может автор запроса в исходном сообщении.",
        }
        await callback.answer(replies[outcome])

    async def unsupported(message: Message) -> None:
        await send_text(message, "Для подбора пришлите текстовый запрос: уровень, стиль и бюджет.")

    router.message.register(welcome, CommandStart())
    router.message.register(welcome, Command("help"))
    router.message.register(unknown_command, F.text.startswith("/"))
    router.message.register(consult, F.text)
    router.message.register(unsupported)
    router.callback_query.register(feedback, F.data.startswith("fb:"))
    return router


async def run_polling(token: str, config: BotConfig, *, check_only: bool = False) -> None:
    store = FeedbackStore(config.feedback_db)
    async with Bot(token=token).context() as bot:
        me = await bot.get_me()
        webhook = await bot.get_webhook_info()
        if webhook.url:
            raise RuntimeError("у бота уже настроен webhook; для этого запуска нужен бот без webhook")
        print(f"Telegram-бот: https://t.me/{me.username}", flush=True)
        if check_only:
            print("Токен принят Telegram, webhook отсутствует. Проверка завершена.", flush=True)
            return
        dispatcher = Dispatcher()
        dispatcher.include_router(create_router(config, store))
        print("Бот принимает сообщения. Остановка: Ctrl+C.", flush=True)
        await dispatcher.start_polling(
            bot, allowed_updates=dispatcher.resolve_used_update_types(),
            tasks_concurrency_limit=32,
        )
