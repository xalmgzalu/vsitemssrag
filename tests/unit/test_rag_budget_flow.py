"""Бюджетный контракт CLI и eval на границе с БД и генерацией."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.run_enrichment_ab import _retrieve_cases
from evals import run_enrichment_ab as runner
from vsitemssrag.cli.main import build_parser
from vsitemssrag.cli.rag import run_ask
from vsitemssrag.rag.evaluation import grade_answer
from vsitemssrag.rag.constraints import no_budget_matches_answer
from decimal import Decimal


class BudgetFlowTests(unittest.TestCase):
    def test_empty_search_without_budget_does_not_leak_cli_instructions_to_bot(self):
        from vsitemssrag.rag.service import ask_catalog
        with contextlib.ExitStack() as stack:
            for name in ("require_setting", "generation_model", "GeminiEmbeddingClient"):
                stack.enter_context(patch("vsitemssrag.rag.service." + name))
            database = stack.enter_context(patch("vsitemssrag.rag.service.Database.from_env"))
            database.return_value.__enter__.return_value.match_products.return_value = []
            generator = stack.enter_context(patch("vsitemssrag.rag.service.GeminiGenerationClient"))
            result = ask_catalog("Ищу быстрое основание")
        generator.assert_not_called()
        self.assertEqual(result.origin, "no_search_results")
        self.assertEqual(result.cited_products, [])
        self.assertNotIn("embed", result.text)

    def test_runner_completes_and_records_deterministic_empty_budget_answers(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            case = {"id": "beginner_control_budget", "question": "Основание до 1000 руб", "max_price_rub": 1000}
            cases = root / "cases.json"
            cases.write_text(json.dumps([case]))
            evidence = root / "evidence.json"
            evidence.write_text('{"products": []}')
            stack.enter_context(patch("sys.argv", ["eval"]))
            args = runner.arguments()
            args.cases, args.evidence_file = cases, evidence
            args.output, args.generation_delay = root / "results.json", 0
            stack.enter_context(patch.object(runner, "arguments", return_value=args))
            stack.enter_context(patch.object(runner, "load_env"))
            stack.enter_context(patch.object(runner, "require_setting", return_value="test"))
            stack.enter_context(patch.object(runner, "_retrieve_cases", return_value=[{
                "id": case["id"], "question": case["question"], "products": [],
                "enriched_product_ids": [], "conditions": {"baseline": [], "enriched": []},
            }]))
            generator = stack.enter_context(patch.object(runner, "GeminiGenerationClient"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(runner.main(), 0)
            payload = json.loads(args.output.read_text())
            generator.return_value.__enter__.return_value.generate.assert_not_called()
        self.assertEqual(payload["status"], "completed")
        for condition in ("baseline", "enriched"):
            self.assertEqual(payload["summary"][condition]["passed_generations"], 3)
            for run in payload["cases"][0]["conditions"][condition]:
                self.assertEqual(run["answer_origin"], "deterministic_no_budget_matches")
                self.assertNotIn("[1]", run["answer"])

    def test_cli_handles_empty_budget_without_generation_or_sources(self):
        for candidates in ([], [{"price": 1001}], [{"price": None}]):
            with self.subTest(candidates=candidates), contextlib.ExitStack() as stack:
                stack.enter_context(patch("vsitemssrag.rag.service.require_setting", return_value="test"))
                stack.enter_context(patch("vsitemssrag.rag.service.generation_model", return_value="test"))
                stack.enter_context(patch("vsitemssrag.rag.service.GeminiEmbeddingClient"))
                database = stack.enter_context(patch("vsitemssrag.rag.service.Database.from_env"))
                database.return_value.__enter__.return_value.match_products.return_value = candidates
                generator = stack.enter_context(patch("vsitemssrag.rag.service.GeminiGenerationClient"))
                output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                args = build_parser().parse_args(["ask", "Основание до 1000 руб"])
                self.assertEqual(run_ask(args), 0)
                self.assertEqual(output.getvalue().strip(), no_budget_matches_answer(Decimal(1000)))
                generator.assert_not_called()
                self.assertNotIn("Источники", output.getvalue())

    def test_cli_passes_only_known_prices_in_budget_to_advisor(self):
        candidates = [
            {"name": "Expensive", "price": 1001},
            {"name": "Unknown", "price": None},
            {"name": "Boundary", "price": 1000, "source_url": "https://example.test/1"},
            {"name": "Cheaper", "price": 500, "source_url": "https://example.test/2"},
        ]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("vsitemssrag.rag.service.require_setting", return_value="test"))
            stack.enter_context(patch("vsitemssrag.rag.service.generation_model", return_value="test"))
            stack.enter_context(patch("vsitemssrag.rag.service.GeminiEmbeddingClient"))
            database = stack.enter_context(patch("vsitemssrag.rag.service.Database.from_env"))
            database.return_value.__enter__.return_value.match_products.return_value = candidates
            stack.enter_context(patch("vsitemssrag.rag.service.GeminiGenerationClient"))
            advisor = stack.enter_context(patch("vsitemssrag.rag.service.generate_advisor_answer", return_value="Варианты [1, 2]."))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(run_ask(build_parser().parse_args(["ask", "Основание до 1000 руб"])), 0)
            self.assertEqual([p["price"] for p in advisor.call_args.args[2]], [1000, 500])
            self.assertNotIn("Expensive", output.getvalue())
            self.assertNotIn("Unknown", output.getvalue())

    def test_eval_keeps_empty_budget_outcome_and_grades_it(self):
        case = {"id": "beginner_control_budget", "question": "Основание до 1000 руб", "max_price_rub": 1000}
        with patch("evals.run_enrichment_ab.GeminiEmbeddingClient"), patch("evals.run_enrichment_ab.Database.from_env") as database, contextlib.redirect_stdout(io.StringIO()):
            database.return_value.__enter__.return_value.match_products.return_value = [{"price": 1001}]
            results = _retrieve_cases([case], api_key="test", timeout=1, limit=6, evidence_by_id={})
        self.assertEqual(results[0]["products"], [])
        self.assertEqual(results[0]["enriched_product_ids"], [])
        answer = no_budget_matches_answer(Decimal(1000))
        self.assertTrue(grade_answer(case, answer, [])["passed"])
        self.assertFalse(grade_answer(case, "Дорогой вариант [1].", [])["passed"])


if __name__ == "__main__":
    unittest.main()
