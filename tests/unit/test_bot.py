"""Реальный Dispatcher aiogram с транспортом без Telegram/API."""
import asyncio
import contextlib
import io
import logging
import os
import sys
import tempfile
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery, GetMe, SendChatAction, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from vsitemssrag.bot.bot import BotConfig, create_router, send_text
from vsitemssrag.bot.feedback import FeedbackStore
from vsitemssrag.bot.presentation import answer_keyboard, is_catalog_url, split_text, telegram_html
from vsitemssrag.cli.main import build_parser
from vsitemssrag.cli.bot import SecretRedactingFormatter, run_bot
from vsitemssrag.rag.service import AdviceResult

TEST_TOKEN = "123456:" + "A" * 35


class FakeTelegramSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.messages = []
        self.reject_html_once = False

    async def close(self):
        pass

    async def stream_content(self, *args, **kwargs):
        yield b""

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        if isinstance(method, GetMe):
            return User(id=123456, is_bot=True, first_name="Test", username="test_advisor_bot")
        if isinstance(method, SendMessage):
            if self.reject_html_once and method.parse_mode == "HTML":
                self.reject_html_once = False
                raise TelegramBadRequest(method=method, message="can't parse entities")
            sent = Message(
                message_id=100 + len(self.messages), date=datetime.now(UTC),
                chat=Chat(id=int(method.chat_id), type="private"),
                text=method.text, reply_markup=method.reply_markup,
            ).as_(bot)
            self.messages.append(sent)
            return sent
        if isinstance(method, (SendChatAction, AnswerCallbackQuery)):
            return True
        raise AssertionError(type(method).__name__)


def incoming(text, user_id=7, update_id=1):
    return Update(update_id=update_id, message=Message(
        message_id=update_id, date=datetime.now(UTC), text=text,
        chat=Chat(id=10, type="private"),
        from_user=User(id=user_id, is_bot=False, first_name="Player"),
    ))


class PresentationTests(unittest.TestCase):
    def test_splits_long_unicode_without_losing_text_or_breaking_utf16_limit(self):
        text = ("**Модель** <unsafe> & 🏓\n" * 500) + "Конец"
        chunks = split_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(c.encode("utf-16-le")) // 2 <= 3500 for c in chunks))

    def test_html_is_escaped_and_only_safe_bold_is_added(self):
        self.assertEqual(telegram_html('**Модель** <a href="evil">&'), '<b>Модель</b> &lt;a href="evil"&gt;&amp;')

    def test_keyboard_uses_only_cited_catalog_products_and_keeps_reference_numbers(self):
        result = AdviceResult("Вариант [2].", [
            {"name": "Unused", "source_url": "https://www.vistasport.ru/catalog/unused.html"},
            {"name": "Chosen", "source_url": "https://www.vistasport.ru/catalog/chosen.html"},
        ])
        keyboard = answer_keyboard(result, "a" * 32)
        self.assertEqual(len(keyboard.inline_keyboard), 2)
        self.assertEqual(keyboard.inline_keyboard[0][0].text, "[2] Chosen")
        self.assertTrue(keyboard.inline_keyboard[0][0].url.endswith("chosen.html"))
        self.assertTrue(all(len(b.callback_data.encode()) <= 64 for b in keyboard.inline_keyboard[-1]))

    def test_no_product_buttons_for_clarification_or_empty_budget(self):
        self.assertEqual(len(answer_keyboard(AdviceResult("Уточните бюджет?"), "a" * 32).inline_keyboard), 1)

    def test_rejects_non_catalog_or_spoofed_urls(self):
        for value in ("javascript:alert(1)", "https://evil.test/catalog/a", "https://vistasport.ru.evil.test/catalog/a", "https://user@vistasport.ru/catalog/a", "http://vistasport.ru/catalog/a", "https://vistasport.ru:bad/catalog/a"):
            self.assertFalse(is_catalog_url(value), value)
        self.assertTrue(is_catalog_url("https://www.vistasport.ru/catalog/blades/a.html"))


class BotDispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.config = BotConfig(root / "evidence.json", root / "feedback.sqlite3", max_concurrent=1)
        self.store = FeedbackStore(self.config.feedback_db)
        self.session = FakeTelegramSession()
        self.bot = Bot(TEST_TOKEN, session=self.session)
        self.result = AdviceResult("**Модель** <unsafe> & [2].", [
            {"name": "Unused", "source_url": "https://www.vistasport.ru/catalog/unused.html"},
            {"name": "Chosen", "source_url": "https://www.vistasport.ru/catalog/chosen.html"},
        ])
        self.advisor = Mock(return_value=self.result)
        self.dispatcher = Dispatcher()
        self.dispatcher.include_router(create_router(self.config, self.store, self.advisor))

    async def asyncTearDown(self):
        await self.bot.session.close()
        self.directory.cleanup()

    async def feed(self, text, user_id=7, update_id=1):
        await self.dispatcher.feed_update(self.bot, incoming(text, user_id, update_id))

    async def vote(self, data, user_id=7, message=None):
        message = message or self.session.messages[-1]
        await self.dispatcher.feed_update(self.bot, Update(update_id=999, callback_query=CallbackQuery(
            id="cb", chat_instance="test", data=data, message=message,
            from_user=User(id=user_id, is_bot=False, first_name="Player"),
        )))

    async def test_start_and_help_do_not_call_rag(self):
        await self.feed("/start")
        await self.feed("/help", update_id=2)
        self.advisor.assert_not_called()
        self.assertIn("6000", self.session.messages[-1].text)

    async def test_unknown_command_is_not_sent_to_model(self):
        await self.feed("/unknown")
        self.advisor.assert_not_called()
        self.assertIn("/help", self.session.messages[-1].text)

    async def test_text_runs_shared_pipeline_and_sends_safe_html_with_buttons(self):
        await self.feed("Новичок, основание до 6000")
        self.advisor.assert_called_once_with("Новичок, основание до 6000", evidence_file=self.config.evidence_file, timeout=30.0)
        answer = self.session.messages[-1]
        self.assertIn("<b>Модель</b> &lt;unsafe&gt; &amp;", answer.text)
        self.assertEqual(answer.reply_markup.inline_keyboard[0][0].text, "[2] Chosen")
        with self.store.connect() as db:
            row = db.execute("select question, message_id, origin from responses").fetchone()
        self.assertEqual(row, ("Новичок, основание до 6000", answer.message_id, "gemini_validated"))

    async def test_feedback_survives_restart_prevents_impersonation_and_deduplicates(self):
        await self.feed("Запрос")
        message = self.session.messages[-1]
        up = message.reply_markup.inline_keyboard[-1][0].callback_data
        down = message.reply_markup.inline_keyboard[-1][1].callback_data
        await self.vote(up, user_id=8, message=message)
        with self.store.connect() as db:
            self.assertEqual(db.execute("select count(*) from feedback").fetchone()[0], 0)
        await self.vote(up, message=message)
        await self.vote(up, message=message)
        # Новые router/store имитируют перезапуск процесса.
        self.dispatcher = Dispatcher()
        self.store = FeedbackStore(self.config.feedback_db)
        self.dispatcher.include_router(create_router(self.config, self.store, self.advisor))
        await self.vote(down, message=message)
        with self.store.connect() as db:
            self.assertEqual(db.execute("select vote from feedback").fetchall(), [(-1,)])
        acks = [m.text for m in self.session.calls if isinstance(m, AnswerCallbackQuery)]
        self.assertIn("Эта оценка уже учтена.", acks)

    async def test_feedback_cannot_be_replayed_on_another_message(self):
        await self.feed("Запрос")
        message = self.session.messages[-1]
        data = message.reply_markup.inline_keyboard[-1][0].callback_data
        await self.vote(data, message=message.model_copy(update={"message_id": 9999}))
        await self.vote("fb:not-a-response:up", message=message)
        with self.store.connect() as db:
            self.assertEqual(db.execute("select count(*) from feedback").fetchone()[0], 0)

    async def test_non_text_update_gets_instruction_without_rag(self):
        original = incoming("temporary")
        update = original.model_copy(update={
            "message": original.message.model_copy(update={"text": None}),
        })
        await self.dispatcher.feed_update(self.bot, update)
        self.advisor.assert_not_called()
        self.assertIn("текстовый запрос", self.session.messages[-1].text)

    async def test_long_answer_has_buttons_only_on_last_chunk(self):
        self.advisor.return_value = AdviceResult("🏓" * 4500)
        await self.feed("Запрос")
        self.assertGreater(len(self.session.messages), 2)
        self.assertTrue(all(m.reply_markup is None for m in self.session.messages[:-1]))
        self.assertIsNotNone(self.session.messages[-1].reply_markup)

    async def test_timeout_is_friendly_and_does_not_poison_next_request(self):
        self.advisor.side_effect = [requests.Timeout("private detail"), self.result]
        with self.assertLogs("vsitemssrag.bot.bot", level="WARNING") as logs:
            await self.feed("Запрос")
        self.assertNotIn("private detail", "\n".join(logs.output))
        self.assertIn("Попробуйте позже", self.session.messages[-1].text)
        self.assertNotIn("private detail", self.session.messages[-1].text)
        await self.feed("Повтор", update_id=2)
        self.assertIsNotNone(self.session.messages[-1].reply_markup)

    async def test_html_parse_failure_retries_as_plain_text(self):
        self.session.reject_html_once = True
        message = incoming("test").message.as_(self.bot)
        await send_text(message, "**Модель** <bad>")
        calls = [m for m in self.session.calls if isinstance(m, SendMessage)]
        self.assertEqual(calls[0].parse_mode, "HTML")
        self.assertIsNone(calls[1].parse_mode)
        self.assertEqual(calls[1].text, "**Модель** <bad>")

    async def test_rag_does_not_block_start_or_allow_duplicate_concurrent_work(self):
        started, release = threading.Event(), threading.Event()
        def slow_advisor(*args, **kwargs):
            started.set()
            release.wait(3)
            return self.result
        self.advisor.side_effect = slow_advisor
        task = asyncio.create_task(self.feed("Медленный запрос"))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            await asyncio.wait_for(self.feed("/start", user_id=8, update_id=2), timeout=1)
            self.assertIn("Салам", self.session.messages[-1].text)
            await self.feed("Повтор запроса", update_id=3)
            self.assertIn("Предыдущий подбор", self.session.messages[-1].text)
            await self.feed("Другой пользователь", user_id=8, update_id=4)
            self.assertIn("другие подборы", self.session.messages[-1].text)
            self.assertEqual(self.advisor.call_count, 1)
        finally:
            release.set()
            await task


class BotCliTests(unittest.TestCase):
    def test_bot_command_defaults_and_check_flag(self):
        args = build_parser().parse_args(["bot", "run", "--check"])
        self.assertEqual(args.bot_command, "run")
        self.assertTrue(args.check)
        self.assertEqual(args.max_concurrent, 2)

    def test_missing_token_returns_clear_error_without_network(self):
        with patch.dict(os.environ, {}, clear=True), patch("vsitemssrag.cli.bot.load_env"), patch("vsitemssrag.settings.load_env"), contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(run_bot(build_parser().parse_args(["bot", "run"])), 1)
        self.assertIn("TELEGRAM_BOT_TOKEN", output.getvalue())

    def test_logs_redact_token_even_in_traceback(self):
        formatter = SecretRedactingFormatter(["secret-value"])
        record = logging.LogRecord("test", logging.ERROR, "test", 1, "url/secret-value", (), None)
        try:
            raise RuntimeError("secret-value")
        except RuntimeError:
            record.exc_info = sys.exc_info()
        self.assertNotIn("secret-value", formatter.format(record))


if __name__ == "__main__":
    unittest.main()
