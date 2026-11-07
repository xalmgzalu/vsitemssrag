import unittest

from vsitemssrag.rag.validation import validate_generated_answer


class RagValidationTests(unittest.TestCase):
    def setUp(self):
        self.products = [
            {
                "name": "Defensive",
                "brand": "A",
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

    def test_accepts_exact_code_rendered_evidence_line(self):
        answer = (
            "Положительный опыт использования с длинными шипами описан "
            "в 1 из 2 просмотренных отзывов [1]."
        )

        self.assertEqual(validate_generated_answer(answer, self.products), [])

    def test_rejects_changed_denominator(self):
        answer = (
            "Положительный опыт использования с длинными шипами описан "
            "в 1 из 5 просмотренных отзывов [1]."
        )

        errors = validate_generated_answer(answer, self.products)

        self.assertTrue(any("1 из 5" in error for error in errors))

    def test_rejects_valid_pair_attached_to_wrong_signal(self):
        products = [
            {
                "name": "Blade",
                "brand": "A",
                "community_evidence": {
                    "reviews_sampled": 5,
                    "mentions": {
                        "control": {"review_count": 4},
                        "soft_feel": {"review_count": 1},
                    },
                },
            }
        ]

        errors = validate_generated_answer(
            "Мягкость упоминается в 4 из 5 просмотренных отзывов [1].",
            products,
        )

        self.assertTrue(any("не соответствует" in error for error in errors))

    def test_rejects_compatibility_overclaim(self):
        errors = validate_generated_answer(
            "Основание совместимо с длинными шипами [1].",
            self.products,
        )

        self.assertTrue(any("совместимость" in error for error in errors))

    def test_long_pips_query_requires_product_and_exact_positive_line(self):
        errors = validate_generated_answer(
            "Уточните бюджет?",
            self.products,
            question="Ищу основание под длинные шипы",
        )

        self.assertTrue(any("не предложено" in error for error in errors))
        self.assertTrue(any("не использована точная строка" in error for error in errors))

    def test_accepts_long_pips_defensive_option_with_disclaimer(self):
        products = [
            {
                "name": "Defensive",
                "brand": "A",
                "community_evidence": {
                    "reviews_sampled": 2,
                    "relations": {
                        "long_pips": {
                            "positive_review_count": 0,
                            "negative_review_count": 0,
                            "unclear_review_count": 1,
                        }
                    },
                },
            }
        ]
        answer = (
            "Защитное основание [1]. Положительный опыт использования с "
            "длинными шипами в просмотренной выборке не зафиксирован [1]."
        )

        self.assertEqual(
            validate_generated_answer(
                answer,
                products,
                question="Ищу основание под длинные шипы",
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
