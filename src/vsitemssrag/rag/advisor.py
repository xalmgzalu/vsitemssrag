"""Общий безопасный generation-проход для CLI и eval."""

from __future__ import annotations

import time
import json
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from vsitemssrag.rag.constraints import (
    CLARIFYING_QUESTION,
    extract_max_price_rub,
    no_budget_matches_answer,
    requires_clarification,
)
from vsitemssrag.rag.context import (
    RAG_SYSTEM_INSTRUCTION, build_rag_prompt, render_community_evidence,
)
from vsitemssrag.rag.gemini import GeminiGenerationClient, GeminiGenerationError
from vsitemssrag.rag.fallback import build_catalog_fallback
from vsitemssrag.rag.validation import (
    answer_citations,
    validate_model_statistics,
    validate_generated_answer,
)


DEFAULT_DIAGNOSTICS_PATH = Path("data/exports/rag_validation_rejections.jsonl")


class AdvisorAnswer(str):
    """Совместимый со строкой ответ с явным происхождением для eval."""
    origin: str

    def __new__(cls, value: str, origin: str):
        result = super().__new__(cls, value)
        result.origin = origin
        return result


def _log_rejection(
    path: Path, *, question: str, answer: str, final_answer: str,
    errors: list[str], attempt: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "question": question, "model_answer": answer, "final_answer": final_answer,
        "errors": errors, "attempt": attempt,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_advisor_answer(
    client: GeminiGenerationClient,
    question: str,
    products: Sequence[Mapping[str, Any]],
    *,
    max_output_tokens: int = 1200,
    temperature: float | None = None,
    validation_attempts: int = 3,
    correction_delay: float = 0.0,
    diagnostics_path: Path = DEFAULT_DIAGNOSTICS_PATH,
    fallback_on_rejection: bool = True,
) -> str:
    if requires_clarification(question):
        return CLARIFYING_QUESTION
    max_price = extract_max_price_rub(question)
    if not products and max_price is not None:
        return no_budget_matches_answer(max_price)
    if validation_attempts <= 0 or correction_delay < 0:
        raise ValueError("некорректные параметры validation")

    base_prompt = build_rag_prompt(question, products)
    prompt = base_prompt
    last_errors: list[str] = []
    for attempt in range(1, validation_attempts + 1):
        answer = client.generate(
            prompt,
            system_instruction=RAG_SYSTEM_INSTRUCTION,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        ).strip()
        # Числа из модели отклоняются до присоединения доверенного блока.
        last_errors = validate_model_statistics(answer)
        last_errors += validate_generated_answer(
            answer, products, question=question, require_evidence=False,
        )
        final_answer = answer
        if not last_errors:
            evidence = render_community_evidence(products, answer_citations(answer))
            if evidence:
                final_answer += "\n\n" + evidence
            last_errors = validate_generated_answer(final_answer, products, question=question)
        if not last_errors:
            return final_answer
        _log_rejection(
            diagnostics_path, question=question, answer=answer,
            final_answer=final_answer, errors=last_errors, attempt=attempt,
        )
        if attempt < validation_attempts:
            if correction_delay:
                time.sleep(correction_delay)
            prompt = (
                f"{base_prompt}\n\n"
                "PREVIOUS_ANSWER_REJECTED:\n"
                f"{answer}\n\n"
                "VALIDATION_ERRORS:\n- "
                + "\n- ".join(last_errors)
                + "\n\nПерепиши рекомендацию полностью, устрани ошибки. "
                "Не включай числа, проценты или оценки частоты из отзывов. "
                "Точные агрегаты для выбранных товаров добавит программа."
            )

    if fallback_on_rejection:
        fallback = build_catalog_fallback(question, products, answer_citations(answer))
        fallback_errors = validate_generated_answer(
            fallback, products,
            question=question if answer_citations(fallback) else None,
        )
        if not fallback_errors:
            return AdvisorAnswer(fallback, "deterministic_catalog_fallback")
        _log_rejection(
            diagnostics_path, question=question, answer=answer, final_answer=fallback,
            errors=fallback_errors, attempt=validation_attempts + 1,
        )
    raise GeminiGenerationError(
        "ответ отклонён grounding-валидатором: " + "; ".join(last_errors)
    )
