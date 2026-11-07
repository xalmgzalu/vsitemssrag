import unittest

from vsitemssrag.rag.evaluation import answer_citations, grade_answer


class RagEvaluationTests(unittest.TestCase):
    def test_accepts_noncomparability_synonym_after_long_scale_description(self):
        answer = (
            "Варианты [1] и [2]. Числовые показатели скорости и контроля у разных "
            "брендов указаны по собственным шкалам производителей, поэтому их "
            "нельзя напрямую сопоставлять как единую шкалу."
        )
        products = [{"name": "A Carbon", "brand": "A"}, {"name": "B Carbon", "brand": "B"}]
        self.assertTrue(grade_answer({"id": "fast_attack_carbon"}, answer, products)["passed"])

    def test_extracts_grouped_citations(self):
        self.assertEqual(answer_citations("Варианты [1, 3, 6] и [2]."), [1, 2, 3, 6])

    def test_accepts_exact_long_pips_aggregate(self):
        case = {"id": "defense_long_pips"}
        products = [
            {
                "name": "Defensive",
                "characteristics": {"Тип основания": "DEF"},
                "community_evidence": {
                    "reviews_sampled": 5,
                    "relations": {
                        "long_pips": {
                            "positive_review_count": 2,
                            "negative_review_count": 0,
                            "unclear_review_count": 0,
                        }
                    },
                },
            }
        ]

        grade = grade_answer(
            case,
            "Положительный опыт использования с длинными шипами описан в 2 из 5 просмотренных отзывов [1].",
            products,
        )

        self.assertTrue(grade["passed"])

    def test_rejects_unqualified_long_pips_claim(self):
        grade = grade_answer(
            {"id": "defense_long_pips"},
            "Основание точно подходит под длинные шипы [1].",
            [{"name": "Defensive"}],
        )

        self.assertFalse(grade["passed"])

    def test_requires_cross_brand_scale_caveat(self):
        products = [
            {"name": "A Carbon", "brand": "A"},
            {"name": "B Carbon", "brand": "B"},
        ]
        case = {"id": "fast_attack_carbon"}

        failed = grade_answer(case, "Быстрые варианты [1] и [2].", products)
        passed = grade_answer(
            case,
            "Варианты [1] и [2]; шкалы производителей различаются, поэтому их нельзя напрямую сравнивать.",
            products,
        )

        self.assertFalse(failed["passed"])
        self.assertTrue(passed["passed"])

    def test_accepts_reverse_order_insufficient_evidence_phrase(self):
        grade = grade_answer(
            {"id": "soft_inner_carbon"},
            "Вариант [1]. Прямых подтверждений мягкости в данных нет.",
            [{"name": "Inner Carbon"}],
        )

        self.assertTrue(grade["passed"])

    def test_accepts_explicit_composite_marker_for_carbon_case(self):
        grade = grade_answer(
            {"id": "fast_attack_carbon"},
            "Быстрый атакующий вариант с композитом [1].",
            [{"name": "OFF+ Blade", "characteristics": {"Конструкция": "с композитом"}}],
        )

        self.assertTrue(grade["passed"])

    def test_rejects_highest_speed_superlative(self):
        grade = grade_answer(
            {"id": "fast_attack_carbon"},
            "У модели самая высокая скорость [1].",
            [{"name": "Carbon OFF+"}],
        )

        self.assertFalse(grade["passed"])

    def test_budget_check_uses_cited_product_price(self):
        grade = grade_answer(
            {"id": "beginner_control_budget", "max_price_rub": 6000},
            "Контрольный вариант [2].",
            [{"price": 5000}, {"price": 7000}],
        )

        self.assertFalse(grade["passed"])


if __name__ == "__main__":
    unittest.main()
