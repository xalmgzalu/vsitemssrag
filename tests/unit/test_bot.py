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
from decimal import Decimal
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
from vsitemssrag.bot.presentation import (
    answer_keyboard, format_price, format_prices, is_catalog_url, render_answer,
    split_text, telegram_html,
)
from vsitemssrag.cli.main import build_parser
from vsitemssrag.cli.bot import SecretRedactingFormatter, run_bot
from vsitemssrag.rag.service import AdviceResult
from vsitemssrag.rag.context import render_community_evidence
from vsitemssrag.rag.fallback import build_catalog_fallback
from vsitemssrag.rag.validation import answer_citations

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

    def test_keyboard_uses_only_cited_catalog_products_with_display_numbers(self):
        result = AdviceResult("Вариант [2].", [
            {"name": "Unused", "source_url": "https://www.vistasport.ru/catalog/unused.html"},
            {"name": "Chosen", "source_url": "https://www.vistasport.ru/catalog/chosen.html"},
        ])
        keyboard = answer_keyboard(result, "a" * 32)
        self.assertEqual(len(keyboard.inline_keyboard), 2)
        self.assertEqual(keyboard.inline_keyboard[0][0].text, "1️⃣ Chosen")
        self.assertTrue(keyboard.inline_keyboard[0][0].url.endswith("chosen.html"))
        self.assertTrue(all(len(b.callback_data.encode()) <= 64 for b in keyboard.inline_keyboard[-1]))

    def test_citations_keep_first_appearance_including_groups_and_duplicates(self):
        self.assertEqual(answer_citations("Сначала [2, 1, 2], затем [3] и [1]."), [2, 1, 3])

    def test_every_card_matches_button_even_for_reversed_and_grouped_references(self):
        products = [
            {"name": "Первый", "price": "2755.00", "source_url": "https://vistasport.ru/catalog/one"},
            {"name": "Второй", "price": "6960.00", "source_url": "https://vistasport.ru/catalog/two"},
            {"name": "Третий", "price": None, "source_url": "https://vistasport.ru/catalog/three"},
        ]
        for text, expected in (
            ("1. Второй [2].\n2. Первый [1].\nСравните [2, 1].", [2, 1]),
            ("Сравните [3, 2, 3] и [1].", [3, 2, 1]),
            ("Третий [3]. Третий [3].", [3]),
        ):
            with self.subTest(text=text):
                result = AdviceResult(text, products)
                rendered = render_answer(result)
                buttons = answer_keyboard(result, "a" * 32).inline_keyboard[:-1]
                self.assertEqual([ref for ref, _ in result.cited_products], expected)
                for index, (ref, row) in enumerate(zip(expected, buttons), 1):
                    marker = f"{index}️⃣"
                    product = products[ref - 1]
                    self.assertIn(f"{marker} **{product['name']} — ", rendered)
                    self.assertEqual(row[0].text, f"{marker} {product['name']}")
                    self.assertEqual(row[0].url, product['source_url'])
                self.assertEqual(answer_citations(rendered), [])

    def test_prices_are_grouped_without_kopecks_and_only_currency_amounts_change(self):
        cases = {
            "6960.00 RUB": "6 960 ₽", "2755.00 руб.": "2 755 ₽",
            "12 345,00 ₽": "12 345 ₽", "1\u202f234,00 RUB": "1 234 ₽",
            "999.49 RUB": "999 ₽", "999.50 RUB": "1 000 ₽",
            "0 руб.": "0 ₽", "5.7 мм, 5+2 слоёв, контроль 8+": "5.7 мм, 5+2 слоёв, контроль 8+",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(format_prices(raw), expected)
        self.assertEqual(format_price(Decimal("6960.00")), "6 960 ₽")
        for invalid in (None, "NaN", "Infinity", -1, "unknown"):
            self.assertEqual(format_price(invalid), "цена не указана")

    def test_citation_spam_is_removed_and_catalog_fields_form_card(self):
        result = AdviceResult(
            "Вот вариант:\n1. **DONIC Original Dotec ALL** [2] — основание без композита [2] типа ALL [2] с ценой 6960.00 RUB.",
            [{"name": "Unused"}, {"name": "DONIC Original Dotec ALL", "price": Decimal("6960"),
             "characteristics": {"Тип основания": "ALL", "Количество слоев": "5", "Конструкция": "Без композита"}}],
        )
        rendered = render_answer(result)
        self.assertIn("1️⃣ **DONIC Original Dotec ALL — 6 960 ₽**", rendered)
        self.assertIn("• Тип и слои: ALL; 5 слоёв", rendered)
        self.assertIn("• Особенности: основание без композита типа ALL с ценой 6 960 ₽.", rendered)
        self.assertNotIn("[2]", rendered)
        self.assertNotIn("RUB", rendered)
        self.assertIn("[2]", result.text)  # Валидация/аудит используют исходный текст.

    def test_scale_caveat_moves_to_footer_but_evidence_and_limitations_stay(self):
        products = [{"name": "Защитное", "price": 4176, "community_evidence": {
            "reviews_sampled": 5, "mentions": {"control": {"review_count": 4}},
            "source_url": "https://reviews.test/blade",
        }}]
        text = (
            "Вот вариант. Обратите внимание, что шкалы производителей могут различаться.\n"
            "Защитное [1] — тип DEF [1]. Положительный опыт использования с длинными шипами не подтверждён [1].\n"
            "Прямого подтверждения мягкости нет."
        )
        result = AdviceResult(text + "\n\n" + render_community_evidence(products, [1]), products)
        rendered = render_answer(result)
        self.assertLess(rendered.index("1️⃣"), rendered.index("ℹ️ Шкалы"))
        self.assertIn("Контроль упоминается в 4 из 5 просмотренных отзывов.", rendered)
        self.assertIn("Источник отзывов: https://reviews.test/blade", rendered)
        self.assertIn("не доказывают пригодность", rendered)
        self.assertIn("Положительный опыт использования с длинными шипами не подтверждён", rendered)
        self.assertIn("Прямого подтверждения мягкости нет.", rendered)
        self.assertEqual(answer_citations(rendered), [])

    def test_unsafe_url_never_shifts_first_button_to_another_card(self):
        result = AdviceResult("Варианты [2] и [1].", [
            {"name": "Safe", "source_url": "https://vistasport.ru/catalog/safe"},
            {"name": "Unsafe", "source_url": "https://evil.test/catalog/a"},
        ])
        keyboard = answer_keyboard(result, "a" * 32)
        self.assertEqual(len(keyboard.inline_keyboard), 1)
        self.assertTrue(all(b.url is None for b in keyboard.inline_keyboard[0]))

    def test_mixed_scale_caveat_keeps_facts_and_correct_display_references(self):
        result = AdviceResult(
            "Шкалы производителей различаются, а мягкость модели [2] не подтверждена.\n"
            "Варианты [2, 1].",
            [{"name": "A"}, {"name": "B"}],
        )
        rendered = render_answer(result)
        self.assertIn("мягкость модели (вариант 1) не подтверждена.", rendered)
        self.assertGreater(rendered.index("ℹ️"), rendered.index("2️⃣"))
        self.assertNotIn("[2]", rendered)

    def test_fallback_keeps_catalog_numbers_and_evidence_for_each_correct_card(self):
        products = [
            {"name": "A", "price": "2755.00", "brand": "A", "characteristics": {"Скорость": "8+"},
             "community_evidence": {"reviews_sampled": 5, "mentions": {"control": {"review_count": 2}}}},
            {"name": "B", "price": "6960.00", "brand": "B", "characteristics": {"Скорость": "9"},
             "community_evidence": {"reviews_sampled": 4, "mentions": {"control": {"review_count": 3}}}},
        ]
        raw = build_catalog_fallback("Хочу мягкое основание", products, [2, 1])
        rendered = render_answer(AdviceResult(raw, products, "deterministic_catalog_fallback"))
        first, second = rendered.split("2️⃣", 1)
        self.assertIn("1️⃣ **B — 6 960 ₽**", first)
        self.assertIn("3 из 4", first)
        self.assertNotIn("2 из 5", first)
        self.assertIn("2 из 5", second)
        self.assertIn("Скорость: 8+.", second)
        self.assertIn("Прямого подтверждения мягкости", rendered)

    def test_baseline_does_not_gain_an_unvalidated_evidence_section(self):
        result = AdviceResult("Вариант [1].", [{"name": "A", "community_evidence": {
            "reviews_sampled": 5, "mentions": {"control": {"review_count": 2}},
        }}])
        self.assertNotIn("2 из 5", render_answer(result))

    def test_name_prefix_does_not_remove_part_of_characteristic(self):
        result = AdviceResult("ALL [1], без композита [1].", [{"name": "A"}])
        self.assertIn("• Особенности: ALL, без композита.", render_answer(result))

    def test_uncited_bullet_continuations_stay_with_their_product(self):
        result = AdviceResult(
            "1. **Второй** [2]\n - Цена: 6960.00 RUB\n - Мягкость не подтверждена.\n\n"
            "2. **Первый** [1]\n - Цена: 2755.00 руб.\n\nОбщее сравнение.",
            [{"name": "Первый", "price": 2755}, {"name": "Второй", "price": 6960}],
        )
        first, second = render_answer(result).split("2️⃣", 1)
        self.assertIn("Цена: 6 960 ₽", first)
        self.assertIn("Мягкость не подтверждена.", first)
        self.assertNotIn("Цена: 2 755 ₽", first)
        self.assertIn("Цена: 2 755 ₽", second)
        self.assertIn("\n\nОбщее сравнение.", second)

    def test_no_cards_or_presentation_rewrite_without_cited_products(self):
        for text in ("Уточните бюджет?", "По вашему запросу не найдены варианты до 1000 ₽."):
            self.assertEqual(render_answer(AdviceResult(text)), text)

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
        self.assertIn("<b>Chosen — цена не указана</b>", answer.text)
        self.assertIn("Модель &lt;unsafe&gt; &amp;", answer.text)
        self.assertEqual(answer.reply_markup.inline_keyboard[0][0].text, "1️⃣ Chosen")
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

    async def test_dispatcher_sends_numbered_cards_and_matching_buttons_in_model_order(self):
        self.advisor.return_value = AdviceResult(
            "Шкалы производителей различаются.\n"
            "1. Chosen [2] — без композита [2], 6960.00 RUB.\n"
            "2. Unused [1] — 2755.00 руб.",
            [dict(self.result.products[0], price="2755.00"), dict(self.result.products[1], price="6960.00")],
        )
        await self.feed("Запрос")
        sent = self.session.messages[-1]
        self.assertLess(sent.text.index("1️⃣ <b>Chosen — 6 960 ₽</b>"), sent.text.index("2️⃣ <b>Unused — 2 755 ₽</b>"))
        self.assertLess(sent.text.index("2️⃣"), sent.text.index("ℹ️"))
        buttons = sent.reply_markup.inline_keyboard
        self.assertEqual(buttons[0][0].url, self.result.products[1]['source_url'])
        self.assertEqual(buttons[1][0].url, self.result.products[0]['source_url'])
        self.assertEqual(answer_citations(sent.text), [])
        with self.store.connect() as db:
            self.assertEqual(db.execute("select answer from responses").fetchone()[0], self.advisor.return_value.text)

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
