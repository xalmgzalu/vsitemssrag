"""Пересчитать сохранённые ответы актуальной версией детерминированного grader."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vsitemssrag.cli.rag import attach_community_evidence, load_community_evidence
from vsitemssrag.rag.evaluation import GRADER_VERSION, grade_answer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "exports" / "enrichment_ab_results.json"
DEFAULT_OUTPUT = ROOT / "data" / "exports" / "enrichment_ab_results_regraded.json"
DEFAULT_CASES = ROOT / "evals" / "rag_cases.json"
DEFAULT_EVIDENCE = ROOT / "data" / "exports" / "review_enrichment_pilot.json"


def _summary(cases: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    runs = [run for case in cases for run in case["conditions"][condition]]
    passed = sum(run["grade"]["passed"] for run in runs)
    return {
        "passed_generations": passed,
        "total_generations": len(runs),
        "pass_rate": round(passed / len(runs), 4) if runs else 0.0,
        "per_case": {
            case["id"]: {
                "passed": sum(
                    run["grade"]["passed"]
                    for run in case["conditions"][condition]
                ),
                "total": len(case["conditions"][condition]),
            }
            for case in cases
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def regrade_payload(
    source: dict[str, Any],
    cases: list[dict[str, Any]],
    evidence_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    cases_by_id = {case["id"]: case for case in cases}
    payload["original_summary"] = payload.get("summary")

    for case_result in payload["cases"]:
        case = cases_by_id[case_result["id"]]
        baseline_products = case_result["products"]
        enriched_products = attach_community_evidence(
            baseline_products,
            evidence_by_id,
        )
        for condition, products in (
            ("baseline", baseline_products),
            ("enriched", enriched_products),
        ):
            for run in case_result["conditions"][condition]:
                run["grade"] = grade_answer(case, run["answer"], products)

    payload["grader_version"] = GRADER_VERSION
    payload["regraded_at"] = datetime.now(UTC).isoformat()
    payload["summary"] = {
        condition: _summary(payload["cases"], condition)
        for condition in ("baseline", "enriched")
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--evidence-file", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    evidence_by_id = load_community_evidence(args.evidence_file)
    payload = regrade_payload(source, cases, evidence_by_id)
    _atomic_write_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Пересчитанный результат: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
