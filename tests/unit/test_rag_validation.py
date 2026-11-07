import unittest

from vsitemssrag.rag.validation import validate_generated_answer, validate_model_statistics


class RagValidationTests(unittest.TestCase):
    def test_long_pips_requires_catalog_type_not_defensive_name_or_reviews(self):
        answer = "Защитное основание [1]. Положительный опыт использования с длинными шипами не подтверждён."
        products = [{"name": "STIGA Defensive Pro"}]
        errors = validate_generated_answer(answer, products, question="Под длинные шипы")
        self.assertTrue(any("нет явного каталожного типа" in e for e in errors))
        for category in ("DEF", "DEF+"):
            products[0]["characteristics"] = {"Тип основания": category}
            self.assertEqual(validate_generated_answer(answer, products, question="Под длинные шипы"), [])

    def test_soft_feel_question_cannot_be_ignored(self):
        products = [{"name": "Inner Carbon"}]
        errors = validate_generated_answer("Вариант [1], конструкция 5+2.", products, question="Хочу мягкий inner carbon")
        self.assertTrue(any("критерий мягкости" in e for e in errors))
        self.assertEqual(validate_generated_answer("Прямых подтверждений мягкости нет [1].", products, question="Хочу мягкий inner carbon"), [])

    def setUp(self):
        self.products = [
            {
                "name": "Defensive",
                "characteristics": {"Тип основания": "DEF"},
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

    def test_negative_confirmation_is_not_a_compatibility_claim(self):
        for answer in (
            "Защитное основание [1]. Положительный опыт использования с длинными шипами в доступных данных не подтверждён.",
            "Защитное основание [1]. Положительный направленный опыт использования с длинными шипами не подтверждён.",
            "Защитное основание [1]. Совместимость с длинными шипами не подтверждена.",
            "Положительный опыт использования с длинными шипами в доступных данных не подтверждён, так как прямой информации о совместимости в каталоге нет. Защитное основание [1].",
            "Прямых подтверждений использования с длинными шипами нет. Защитное основание [1].",
            "Отзывы не гарантируют совместимость с длинными шипами [1].",
        ):
            with self.subTest(answer=answer):
                errors = validate_generated_answer(answer, [{"name": "DEF"}])
                self.assertFalse(any("превращено в совместимость" in e for e in errors))
        errors = validate_generated_answer(
            "Совместимость с длинными шипами не подтверждена, но основание гарантированно подходит под длинные шипы [1].",
            self.products,
        )
        self.assertTrue(any("превращено в совместимость" in e for e in errors))
        for claim in (
            "Основание совместимо с длинными шипами [1].",
            "Доказана совместимость с длинными шипами [1].",
            "Основание гарантированно подходит для длинных шипов [1].",
        ):
            self.assertTrue(validate_generated_answer(claim, self.products))

    def test_speed_superlative_with_only_one_citation_is_rejected(self):
        errors = validate_generated_answer(
            "Имеет наибольший показатель скорости среди представленных вариантов (10++) [1].",
            self.products,
        )
        self.assertTrue(any("максимальная скорость" in e for e in errors))
        self.assertEqual(
            validate_generated_answer(
                "Учитывая запрос на максимальный контроль, шкалы скорости нельзя напрямую сравнивать. Вариант [1].",
                self.products,
            ), [],
        )

    def test_long_pips_query_requires_product_and_exact_positive_line(self):
        errors = validate_generated_answer(
            "Уточните бюджет?",
            self.products,
            question="Ищу основание под длинные шипы",
        )

        self.assertTrue(any("не предложено" in error for error in errors))
        selected_errors = validate_generated_answer(
            "Защитный вариант [1].", self.products,
            question="Ищу основание под длинные шипы",
        )
        self.assertTrue(any("не использована точная строка" in error for error in selected_errors))

    def test_model_cannot_supply_statistics_in_alternate_formats(self):
        for answer in (
            "80% отзывов отмечают контроль [1].",
            "Контроль отмечают 2/5 авторов [1].",
            "В двух из пяти отзывов есть контроль [1].",
            "Большинство игроков хвалят контроль [1].",
        ):
            with self.subTest(answer=answer):
                self.assertTrue(validate_model_statistics(answer))
        self.assertEqual(validate_model_statistics("В отзывах упоминается контроль [1]."), [])
        self.assertEqual(
            validate_model_statistics("Контроль упоминается в отзывах о DONIC Defplay Classic Senso V3 [2]."),
            [],
        )
        self.assertEqual(
            validate_model_statistics("У модели 5+2 слоя, а в отзывах упоминается контроль [1]."),
            [],
        )

    def test_accepts_long_pips_defensive_option_with_disclaimer(self):
        products = [
            {
                "name": "Defensive",
                "characteristics": {"Тип основания": "DEF"},
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
