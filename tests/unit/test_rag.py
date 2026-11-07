import json
import unittest
from decimal import Decimal
from pathlib import Path

from vsitemssrag.rag.constraints import (
    CLARIFYING_QUESTION,
    extract_max_price_rub,
    products_within_budget,
    requires_clarification,
)
from vsitemssrag.rag.context import RAG_SYSTEM_INSTRUCTION, build_rag_prompt
from vsitemssrag.rag.gemini import (
    GeminiGenerationClient,
    GeminiGenerationError,
    parse_generation_response,
)


class RagContextTests(unittest.TestCase):
    def test_builds_json_context_without_internal_similarity(self):
        prompt = build_rag_prompt(
            "  Что взять новичку до 6000? ",
            [
                {
                    "name": "Основание Test",
                    "brand": "TestBrand",
                    "price": Decimal("5990.00"),
                    "currency": "RUB",
                    "source_url": "https://example.test/blade",
                    "characteristics": {"Контроль": "9"},
                    "similarity": 0.91,
                }
            ],
        )

        question, raw_catalog = prompt.split("\n\nCATALOG_DATA:\n", 1)
        catalog = json.loads(raw_catalog)
        self.assertEqual(question, "USER_QUESTION:\nЧто взять новичку до 6000?")
        self.assertEqual(catalog[0]["ref"], 1)
        self.assertEqual(catalog[0]["price"], "5990.00")
        self.assertEqual(catalog[0]["characteristics"], {"Контроль": "9"})
        self.assertNotIn("similarity", catalog[0])

    def test_rejects_empty_question_or_catalog(self):
        with self.assertRaises(ValueError):
            build_rag_prompt("", [{"name": "Blade"}])
        with self.assertRaises(ValueError):
            build_rag_prompt("Что выбрать?", [])

    def test_system_instruction_treats_catalog_as_untrusted_data(self):
        self.assertIn("недоверенные данные", RAG_SYSTEM_INSTRUCTION)
        self.assertIn("не придумывай", RAG_SYSTEM_INSTRUCTION)
        self.assertIn("не сравнивай числовые показатели", RAG_SYSTEM_INSTRUCTION)
        self.assertIn("субъективный агрегат", RAG_SYSTEM_INSTRUCTION)

    def test_adds_only_aggregated_community_evidence(self):
        prompt = build_rag_prompt(
            "Подойдёт под длинные шипы?",
            [
                {
                    "name": "Test Blade",
                    "source_url": "https://shop.test/blade",
                    "community_evidence": {
                        "source": "Example Reviews",
                        "source_url": "https://reviews.test/blade",
                        "reviews_sampled": 5,
                        "mentions": {
                            "control": {
                                "review_count": 2,
                                "review_urls": ["https://reviews.test/raw"],
                            }
                        },
                        "raw_review_text": "этот текст нельзя передавать",
                    },
                }
            ],
        )

        catalog = json.loads(prompt.split("\n\nCATALOG_DATA:\n", 1)[1])
        evidence = catalog[0]["community_evidence"]
        self.assertIn(
            "Контроль упоминается в 2 из 5 просмотренных отзывов [1].",
            evidence["summary_lines"],
        )
        self.assertNotIn("raw_review_text", evidence)
        self.assertNotIn("review_urls", json.dumps(evidence))


class RagConstraintTests(unittest.TestCase):
    def test_requires_clarification_only_for_generic_request(self):
        self.assertTrue(requires_clarification("Посоветуй хорошее основание"))
        self.assertFalse(
            requires_clarification("Посоветуй быстрое карбоновое основание")
        )
        self.assertEqual(CLARIFYING_QUESTION.count("?"), 1)

    def test_extracts_ruble_budget(self):
        self.assertEqual(extract_max_price_rub("бюджет до 6 000 ₽"), Decimal("6000"))
        self.assertEqual(extract_max_price_rub("не дороже 6,5 тысяч"), Decimal("6500.0"))
        self.assertEqual(extract_max_price_rub("максимум 7к"), Decimal("7000"))
        self.assertIsNone(extract_max_price_rub("хочу контрольное основание"))

    def test_filters_unknown_and_expensive_products(self):
        products = [
            {"name": "Expensive", "price": Decimal("7000")},
            {"name": "Unknown", "price": None},
            {"name": "Fit", "price": Decimal("5990")},
            {"name": "Also fit", "price": 4500},
        ]

        result = products_within_budget(products, Decimal("6000"), limit=2)

        self.assertEqual([product["name"] for product in result], ["Fit", "Also fit"])


class GeminiGenerationResponseTests(unittest.TestCase):
    def test_combines_text_parts(self):
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Первая часть."},
                            {"text": "Вторая часть."},
                        ]
                    }
                }
            ]
        }

        self.assertEqual(
            parse_generation_response(response),
            "Первая часть.\nВторая часть.",
        )

    def test_reports_blocked_prompt(self):
        with self.assertRaisesRegex(GeminiGenerationError, "SAFETY"):
            parse_generation_response({"promptFeedback": {"blockReason": "SAFETY"}})

    def test_sends_grounded_generate_content_request(self):
        class FakeResponse:
            ok = True

            @staticmethod
            def json():
                return {
                    "candidates": [
                        {"content": {"parts": [{"text": "Ответ [1]."}]}}
                    ]
                }

        class FakeSession:
            def __init__(self):
                self.call = None

            def post(self, url, **kwargs):
                self.call = (url, kwargs)
                return FakeResponse()

        session = FakeSession()
        client = GeminiGenerationClient(
            "secret",
            model="gemini-test",
            session=session,
        )

        answer = client.generate(
            "USER_QUESTION: test",
            system_instruction="Используй только каталог",
            max_output_tokens=321,
            temperature=0,
        )

        self.assertEqual(answer, "Ответ [1].")
        url, kwargs = session.call
        self.assertTrue(url.endswith("/gemini-test:generateContent"))
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "secret")
        self.assertEqual(
            kwargs["json"]["generationConfig"]["maxOutputTokens"],
            321,
        )
        self.assertEqual(kwargs["json"]["generationConfig"]["temperature"], 0)
        self.assertEqual(
            kwargs["json"]["systemInstruction"]["parts"][0]["text"],
            "Используй только каталог",
        )


class RagEvaluationFixtureTests(unittest.TestCase):
    def test_cases_have_unique_ids_and_machine_checkable_budgets(self):
        fixture_path = Path(__file__).resolve().parents[2] / "evals" / "rag_cases.json"
        cases = json.loads(fixture_path.read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(cases), 5)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        for case in cases:
            self.assertTrue(case["question"].strip())
            expected_budget = case.get("max_price_rub")
            if expected_budget is not None:
                self.assertEqual(
                    extract_max_price_rub(case["question"]),
                    Decimal(str(expected_budget)),
                )


if __name__ == "__main__":
    unittest.main()
