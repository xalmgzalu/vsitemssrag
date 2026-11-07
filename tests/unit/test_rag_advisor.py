import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from vsitemssrag.rag.gemini import GeminiGenerationError

from vsitemssrag.rag.advisor import generate_advisor_answer
from vsitemssrag.rag.constraints import CLARIFYING_QUESTION


class FakeGenerationClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = 0

    def generate(self, *_args, **_kwargs):
        self.calls += 1
        return next(self.answers)


class RagAdvisorTests(unittest.TestCase):
    def test_repeated_compatibility_claim_uses_safe_catalog_fallback(self):
        products = [
            {"name": "Unknown type", "characteristics": {}},
            {"name": "Defensive", "price": 1000, "currency": "RUB", "characteristics": {"Тип основания": "DEF"},
             "community_evidence": {"reviews_sampled": 5, "mentions": {"control": {"review_count": 2}}, "source_url": "https://reviews.test/def"}},
        ]
        invalid = "Основание гарантированно подходит под длинные шипы [1, 2]."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rejections.jsonl"
            client = FakeGenerationClient([invalid] * 3)
            answer = generate_advisor_answer(client, "Ищу защитное основание под длинные шипы", products, diagnostics_path=path)
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(answer.origin, "deterministic_catalog_fallback")
        self.assertEqual(client.calls, 3)
        self.assertEqual(len(records), 3)
        self.assertNotIn("Unknown type", answer)
        self.assertNotIn("гарантированно", answer)
        self.assertIn("не подтверждён", answer)
        self.assertIn("Контроль упоминается в 2 из 5 просмотренных отзывов [2].", answer)
        self.assertIn("https://reviews.test/def", answer)

    def test_fallback_without_matching_catalog_type_does_not_invent_one(self):
        with tempfile.TemporaryDirectory() as directory:
            answer = generate_advisor_answer(
                FakeGenerationClient(["Совместимо с длинными шипами [1]."]),
                "Под длинные шипы", [{"name": "Unknown"}], validation_attempts=1,
                diagnostics_path=Path(directory) / "rejections.jsonl",
            )
        self.assertEqual(answer.origin, "deterministic_catalog_fallback")
        self.assertNotIn("[1]", answer)
        self.assertIn("В поисковой выборке нет", answer)

    def test_empty_budget_result_does_not_call_generator(self):
        client = FakeGenerationClient([])
        answer = generate_advisor_answer(client, "Основание до 1000 руб", [])
        self.assertIn("не найдены варианты до 1000 ₽", answer)
        self.assertEqual(client.calls, 0)

    def test_generic_request_returns_deterministic_question_without_gemini(self):
        client = FakeGenerationClient([])

        answer = generate_advisor_answer(
            client,
            "Посоветуй хорошее основание",
            [{"name": "Blade"}],
        )

        self.assertEqual(answer, CLARIFYING_QUESTION)
        self.assertEqual(client.calls, 0)

    def test_retries_answer_with_changed_evidence_number(self):
        products = [
            {
                "name": "Defensive",
                "characteristics": {"Тип основания": "DEF"},
                "community_evidence": {
                    "reviews_sampled": 2,
                    "relations": {
                        "long_pips": {
                            "positive_review_count": 1,
                            "negative_review_count": 0,
                            "unclear_review_count": 0,
                        }
                    },
                },
            }
        ]
        invalid = (
            "Положительный опыт использования с длинными шипами описан "
            "в 1 из 5 просмотренных отзывов [1]."
        )
        valid = (
            "Положительный опыт использования с длинными шипами описан "
            "в 1 из 2 просмотренных отзывов [1]."
        )
        client = FakeGenerationClient([invalid, "Защитное основание [1]."])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rejections.jsonl"
            answer = generate_advisor_answer(
                client, "Ищу защитное основание под длинные шипы", products,
                diagnostics_path=path,
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["model_answer"], invalid)
        self.assertTrue(records[0]["errors"])
        self.assertIn(valid, answer)
        self.assertEqual(client.calls, 2)


    def test_evidence_only_for_selected_products_and_preserves_source(self):
        products = [
            {"name": "Chosen", "community_evidence": {
                "reviews_sampled": 5, "mentions": {"speed": {"review_count": 2}},
                "source_url": "https://reviews.test/chosen",
            }},
            {"name": "Unchosen", "community_evidence": {
                "reviews_sampled": 9, "mentions": {"control": {"review_count": 8}},
                "source_url": "https://reviews.test/unchosen",
            }},
        ]
        answer = generate_advisor_answer(FakeGenerationClient(["Карбоновый вариант [1]."]), "Быстрое карбоновое основание", products)
        self.assertIn("Скорость упоминается в 2 из 5 просмотренных отзывов [1].", answer)
        self.assertIn("https://reviews.test/chosen", answer)
        self.assertNotIn("8 из 9", answer)
        self.assertNotIn("Unchosen", answer)
        self.assertNotIn("https://reviews.test/unchosen", answer)

    def test_baseline_does_not_gain_evidence(self):
        answer = generate_advisor_answer(FakeGenerationClient(["Вариант [1]."]), "Быстрое основание", [{"name": "Blade"}])
        self.assertEqual(answer, "Вариант [1].")

    def test_even_correct_model_statistics_are_rejected_and_logged(self):
        invalid = "Контроль упоминается в 2 из 5 просмотренных отзывов [1]."
        products = [{"community_evidence": {"reviews_sampled": 5, "mentions": {"control": {"review_count": 2}}}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rejections.jsonl"
            with self.assertRaises(GeminiGenerationError):
                generate_advisor_answer(FakeGenerationClient([invalid]), "Контрольное основание", products, validation_attempts=1, diagnostics_path=path, fallback_on_rejection=False)
            record = json.loads(path.read_text())
        self.assertEqual(record["attempt"], 1)
        self.assertIn("статистику", record["errors"][0])

    def test_unsupported_claim_rejected_even_with_valid_evidence_block(self):
        products = [{"community_evidence": {"reviews_sampled": 5, "mentions": {"control": {"review_count": 2}}}}]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(GeminiGenerationError):
                generate_advisor_answer(
                    FakeGenerationClient(["Основание гарантированно подходит под длинные шипы [1]."]),
                    "Под длинные шипы", products, validation_attempts=1,
                    diagnostics_path=Path(directory) / "rejections.jsonl", fallback_on_rejection=False,
                )

    def test_final_assembled_answer_is_validated_and_logged(self):
        products = [{"name": "Blade"}]
        invalid_block = "Мягкость упоминается в 4 из 5 просмотренных отзывов [1]."
        with tempfile.TemporaryDirectory() as directory, patch(
            "vsitemssrag.rag.advisor.render_community_evidence", return_value=invalid_block,
        ):
            path = Path(directory) / "rejections.jsonl"
            with self.assertRaises(GeminiGenerationError):
                generate_advisor_answer(FakeGenerationClient(["Вариант [1]. Прямых подтверждений мягкости нет."]), "Мягкое основание", products, validation_attempts=1, diagnostics_path=path, fallback_on_rejection=False)
            record = json.loads(path.read_text())
        self.assertIn(invalid_block, record["final_answer"])
        self.assertTrue(any("не соответствует" in e for e in record["errors"]))


if __name__ == "__main__":
    unittest.main()
