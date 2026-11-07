"""Запустить воспроизводимый A/B-тест агрегатов отзывов на реальном RAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests

from vsitemssrag.cli.rag import attach_community_evidence, load_community_evidence
from vsitemssrag.embeddings import (
    EMBEDDING_MODEL,
    GeminiEmbeddingClient,
    build_search_query,
)
from vsitemssrag.rag import (
    GENERATION_MODEL,
    GeminiGenerationClient,
    GeminiGenerationError,
    extract_max_price_rub,
    generate_advisor_answer,
    requires_clarification,
)
from vsitemssrag.rag.context import RAG_SYSTEM_INSTRUCTION
from vsitemssrag.rag.reranker import RETRIEVAL_VERSION, candidate_pool_limit, select_ranked_products
from vsitemssrag.rag.evaluation import GRADER_VERSION, grade_answer
from vsitemssrag.rag.validation import VALIDATOR_VERSION
from vsitemssrag.settings import load_env, require_setting
from vsitemssrag.storage.postgres import Database


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evals" / "rag_cases.json"
DEFAULT_PROTOCOL = ROOT / "evals" / "enrichment_ab_protocol_v7.json"
DEFAULT_EVIDENCE = ROOT / "data" / "exports" / "review_enrichment_pilot_v2.json"
DEFAULT_OUTPUT = ROOT / "data" / "exports" / "enrichment_ab_results_v7.json"


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return parsed


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--evidence-file", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--limit", type=positive, default=6)
    parser.add_argument("--max-output-tokens", type=positive, default=1200)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--generation-delay",
        type=float,
        help="пауза между генерациями; по умолчанию берётся из protocol",
    )
    parser.add_argument("--generation-attempts", type=positive, default=5)
    parser.add_argument("--retry-wait", type=float, default=45.0)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="начать заново и перезаписать существующий checkpoint",
    )
    parser.add_argument("--model")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _summary(cases: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    runs = [
        run
        for case in cases
        for run in case["conditions"][condition]
    ]
    passed = sum(run["grade"]["passed"] for run in runs)
    per_case = {
        case["id"]: {
            "passed": sum(
                run["grade"]["passed"] for run in case["conditions"][condition]
            ),
            "total": len(case["conditions"][condition]),
        }
        for case in cases
    }
    return {
        "passed_generations": passed,
        "total_generations": len(runs),
        "pass_rate": round(passed / len(runs), 4) if runs else 0.0,
        "per_case": per_case,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _refresh_checkpoint(
    payload: dict[str, Any],
    path: Path,
    *,
    status: str,
    last_error: str | None = None,
) -> None:
    payload["status"] = status
    payload["updated_at"] = datetime.now(UTC).isoformat()
    payload["last_error"] = last_error
    payload["summary"] = {
        condition: _summary(payload["cases"], condition)
        for condition in ("baseline", "enriched")
    }
    _atomic_write_json(path, payload)


def _configuration(
    args: argparse.Namespace,
    *,
    model: str,
    evidence_count: int,
    case_ids: list[str],
    generation_delay: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "embedding_model": EMBEDDING_MODEL,
        "temperature": 0,
        "grader_version": GRADER_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "system_instruction_sha256": hashlib.sha256(
            RAG_SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "repeats": args.repeats,
        "limit": args.limit,
        "generation_delay": generation_delay,
        "system_instruction_identical": True,
        "retrieval_identical": True,
        "retrieval_version": RETRIEVAL_VERSION,
        "retrieval_code_sha256": {
            name: _file_sha256(ROOT / "src" / "vsitemssrag" / "rag" / name)
            for name in ("reranker.py", "popularity.py", "materials.py", "materials_ontology.json", "context.py", "constraints.py")
        },
        "catalog_adapter_sha256": _file_sha256(ROOT / "src" / "vsitemssrag" / "storage" / "postgres.py"),
        "evidence_file": str(args.evidence_file.resolve()),
        "evidence_sha256": _file_sha256(args.evidence_file),
        "cases_sha256": _file_sha256(args.cases),
        "protocol_sha256": _file_sha256(args.protocol),
        "selected_case_ids": case_ids,
        "matched_evidence_products": evidence_count,
    }


def _load_checkpoint(
    path: Path,
    expected_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError(
            "существующий result несовместим; используйте --restart для нового прогона"
        )
    if payload.get("configuration") != expected_configuration:
        raise ValueError(
            "конфигурация checkpoint отличается; продолжите с прежними параметрами "
            "или используйте --restart"
        )
    return payload


def _retrieve_cases(
    cases: list[dict[str, Any]],
    *,
    api_key: str,
    timeout: float,
    limit: int,
    evidence_by_id: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    embeddings: list[list[float]] = []
    with GeminiEmbeddingClient(api_key, delay=1.0, timeout=timeout) as embedder:
        for case_index, case in enumerate(cases, start=1):
            print(f"[{case_index}/{len(cases)}] embedding: {case['id']}", flush=True)
            embeddings.append(embedder.embed(build_search_query(case["question"])))

    results: list[dict[str, Any]] = []
    # БД открывается только после embeddings и закрывается до долгой генерации,
    # поэтому pooler не держит простаивающее соединение несколько минут.
    with Database.from_env() as database:
        for case_index, (case, query_embedding) in enumerate(
            zip(cases, embeddings, strict=True),
            start=1,
        ):
            print(f"[{case_index}/{len(cases)}] retrieval: {case['id']}", flush=True)
            max_price = extract_max_price_rub(case["question"])
            candidates = database.match_products(
                query_embedding,
                model=EMBEDDING_MODEL,
                limit=candidate_pool_limit(limit),
                only_available=True,
            )
            products = select_ranked_products(case["question"], candidates, max_price, limit=limit)
            if not products and max_price is None:
                raise RuntimeError(f"нет retrieval-результатов для {case['id']}")

            enriched_products = attach_community_evidence(products, evidence_by_id)
            results.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "products": _jsonable(products),
                    "enriched_product_ids": [
                        product["product_id"]
                        for product in enriched_products
                        if "community_evidence" in product
                    ],
                    "conditions": {"baseline": [], "enriched": []},
                }
            )
    return results


def _retryable_generation_error(error: Exception) -> bool:
    if isinstance(error, requests.RequestException):
        return True
    message = str(error).casefold()
    return any(
        marker in message
        for marker in ("http 429", "http 500", "http 502", "http 503", "http 504")
    )


def _generate_with_retries(
    generator: GeminiGenerationClient,
    question: str,
    products: list[dict[str, Any]],
    *,
    max_output_tokens: int,
    attempts: int,
    retry_wait: float,
    correction_delay: float,
) -> str:
    for attempt in range(1, attempts + 1):
        try:
            return generate_advisor_answer(
                generator,
                question,
                products,
                max_output_tokens=max_output_tokens,
                temperature=0,
                correction_delay=correction_delay,
            )
        except (GeminiGenerationError, requests.RequestException) as error:
            if attempt == attempts or not _retryable_generation_error(error):
                raise
            wait_seconds = min(retry_wait * attempt, 60.0)
            print(
                f"    временная ошибка Gemini; попытка {attempt}/{attempts}, "
                f"пауза {wait_seconds:g} с",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise AssertionError("цикл generation attempts завершился без результата")


def main() -> int:
    args = arguments()
    if args.timeout <= 0 or args.retry_wait < 0:
        raise ValueError("timeout должен быть > 0, retry-wait должен быть >= 0")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.repeats != protocol["controls"]["repetitions_per_condition"]:
        raise ValueError("--repeats должен совпадать с зафиксированным protocol")
    if args.limit != protocol["controls"]["context_products"]:
        raise ValueError("--limit должен совпадать с зафиксированным protocol")
    generation_delay = (
        args.generation_delay
        if args.generation_delay is not None
        else float(protocol["controls"]["generation_request_interval_seconds"])
    )
    if generation_delay < 0:
        raise ValueError("generation-delay должен быть >= 0")

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"неизвестные case id: {sorted(missing)}")

    load_env()
    api_key = require_setting("GEMINI_API_KEY")
    model = args.model or os.getenv("GEMINI_GENERATION_MODEL", "").strip() or GENERATION_MODEL
    evidence_by_id = load_community_evidence(args.evidence_file)
    case_ids = [case["id"] for case in cases]
    configuration = _configuration(
        args,
        model=model,
        evidence_count=len(evidence_by_id),
        case_ids=case_ids,
        generation_delay=generation_delay,
    )

    if args.output.exists() and not args.restart:
        payload = _load_checkpoint(args.output, configuration)
        print(f"Продолжаю checkpoint: {args.output}", flush=True)
    else:
        results = _retrieve_cases(
            cases,
            api_key=api_key,
            timeout=args.timeout,
            limit=args.limit,
            evidence_by_id=evidence_by_id,
        )
        payload = {
            "schema_version": 2,
            "status": "running",
            "generated_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "last_error": None,
            "protocol": protocol,
            "configuration": configuration,
            "summary": {},
            "cases": results,
        }
        _refresh_checkpoint(payload, args.output, status="running")
        print(f"Retrieval checkpoint: {args.output}", flush=True)

    cases_by_id = {case["id"]: case for case in cases}
    last_generation_finished_at: float | None = None
    with GeminiGenerationClient(
        api_key,
        model=model,
        timeout=args.timeout,
    ) as generator:
        for case_index, case_result in enumerate(payload["cases"], start=1):
            case = cases_by_id[case_result["id"]]
            products = case_result["products"]
            enriched_products = attach_community_evidence(products, evidence_by_id)
            print(f"[{case_index}/{len(cases)}] generation: {case['id']}", flush=True)
            no_budget_matches = (
                not products and extract_max_price_rub(case["question"]) is not None
            )
            deterministic = requires_clarification(case["question"]) or no_budget_matches
            for repeat in range(1, args.repeats + 1):
                for condition, context_products in (
                    ("baseline", products),
                    ("enriched", enriched_products),
                ):
                    completed_repeats = {
                        run["repeat"]
                        for run in case_result["conditions"][condition]
                    }
                    if repeat in completed_repeats:
                        print(
                            f"  generation {repeat}/{args.repeats}: {condition} (checkpoint)",
                            flush=True,
                        )
                        continue

                    if not deterministic and last_generation_finished_at is not None:
                        elapsed = time.monotonic() - last_generation_finished_at
                        if elapsed < generation_delay:
                            time.sleep(generation_delay - elapsed)
                    print(
                        f"  generation {repeat}/{args.repeats}: {condition}",
                        flush=True,
                    )
                    try:
                        answer = _generate_with_retries(
                            generator,
                            case["question"],
                            context_products,
                            max_output_tokens=args.max_output_tokens,
                            attempts=args.generation_attempts,
                            retry_wait=args.retry_wait,
                            correction_delay=generation_delay,
                        )
                    except (GeminiGenerationError, requests.RequestException) as error:
                        last_generation_finished_at = time.monotonic()
                        message = f"{type(error).__name__}: {error}"
                        _refresh_checkpoint(
                            payload,
                            args.output,
                            status="paused",
                            last_error=message,
                        )
                        print(f"Эксперимент приостановлен: {message}", flush=True)
                        print("Повторите ту же команду для продолжения.", flush=True)
                        return 1

                    last_generation_finished_at = time.monotonic()
                    case_result["conditions"][condition].append(
                        {
                            "repeat": repeat,
                            "answer": answer,
                            "answer_origin": (
                                "deterministic_no_budget_matches"
                                if no_budget_matches
                                else "deterministic_clarification"
                                if deterministic
                                else getattr(answer, "origin", "gemini_validated")
                            ),
                            "grade": grade_answer(case, answer, context_products),
                        }
                    )
                    _refresh_checkpoint(payload, args.output, status="running")

    _refresh_checkpoint(payload, args.output, status="completed")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"Полный результат: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
