import unittest

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
        client = FakeGenerationClient([invalid, valid])

        answer = generate_advisor_answer(
            client,
            "Ищу защитное основание под длинные шипы",
            products,
        )

        self.assertEqual(answer, valid)
        self.assertEqual(client.calls, 2)


if __name__ == "__main__":
    unittest.main()
