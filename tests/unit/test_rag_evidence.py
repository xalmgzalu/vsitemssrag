import json
import tempfile
import unittest
from pathlib import Path

from vsitemssrag.cli.rag import (
    attach_community_evidence,
    load_community_evidence,
)


class RagEvidenceTests(unittest.TestCase):
    def test_loader_accepts_only_matched_products(self):
        payload = {
            "products": [
                {
                    "product_id": 1,
                    "match": {"status": "matched", "evidence": {"reviews_sampled": 3}},
                },
                {
                    "product_id": 2,
                    "match": {"status": "candidate", "evidence": {"reviews_sampled": 9}},
                },
                {"product_id": 3, "match": {"status": "unmatched"}},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            evidence = load_community_evidence(path)

        self.assertEqual(evidence, {1: {"reviews_sampled": 3}})

    def test_attaches_evidence_without_mutating_search_results(self):
        products = [{"product_id": 1, "name": "Blade"}]

        enriched = attach_community_evidence(products, {1: {"reviews_sampled": 3}})

        self.assertNotIn("community_evidence", products[0])
        self.assertEqual(enriched[0]["community_evidence"]["reviews_sampled"], 3)


if __name__ == "__main__":
    unittest.main()
