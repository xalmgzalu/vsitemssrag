import tempfile
import unittest
from pathlib import Path

from evals.run_enrichment_ab import (
    _atomic_write_json,
    _load_checkpoint,
    _refresh_checkpoint,
    _generate_with_retries,
)
from vsitemssrag.rag.advisor import AdvisorAnswer
from unittest.mock import patch


class EvalRunnerCheckpointTests(unittest.TestCase):
    def test_generation_wrapper_preserves_fallback_origin(self):
        answer = AdvisorAnswer("По каталогу [1].", "deterministic_catalog_fallback")
        with patch("evals.run_enrichment_ab.generate_advisor_answer", return_value=answer):
            result = _generate_with_retries(
                object(), "Вопрос", [], max_output_tokens=100,
                attempts=1, retry_wait=0, correction_delay=0,
            )
        self.assertEqual(result.origin, "deterministic_catalog_fallback")

    def test_checkpoint_round_trip_and_summary(self):
        configuration = {"model": "test"}
        payload = {
            "schema_version": 2,
            "configuration": configuration,
            "cases": [
                {
                    "id": "case",
                    "conditions": {
                        "baseline": [{"grade": {"passed": True}}],
                        "enriched": [],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"

            _refresh_checkpoint(payload, path, status="running")
            restored = _load_checkpoint(path, configuration)

        self.assertEqual(restored["status"], "running")
        self.assertEqual(restored["summary"]["baseline"]["passed_generations"], 1)
        self.assertEqual(restored["summary"]["enriched"]["total_generations"], 0)

    def test_rejects_different_resume_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            _atomic_write_json(
                path,
                {"schema_version": 2, "configuration": {"model": "old"}},
            )

            with self.assertRaisesRegex(ValueError, "конфигурация checkpoint"):
                _load_checkpoint(path, {"model": "new"})


if __name__ == "__main__":
    unittest.main()
