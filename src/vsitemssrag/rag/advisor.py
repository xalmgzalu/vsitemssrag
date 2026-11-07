"""Общий безопасный generation-проход для CLI и eval."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from vsitemssrag.rag.constraints import CLARIFYING_QUESTION, requires_clarification
from vsitemssrag.rag.context import RAG_SYSTEM_INSTRUCTION, build_rag_prompt
from vsitemssrag.rag.gemini import GeminiGenerationClient, GeminiGenerationError
from vsitemssrag.rag.validation import (
    allowed_community_evidence_lines,
    validate_generated_answer,
)


def generate_advisor_answer(
    client: GeminiGenerationClient,
    question: str,
    products: Sequence[Mapping[str, Any]],
    *,
    max_output_tokens: int = 1200,
    temperature: float | None = None,
    validation_attempts: int = 3,
    correction_delay: float = 0.0,
) -> str:
    if requires_clarification(question):
        return CLARIFYING_QUESTION
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
        last_errors = validate_generated_answer(
            answer,
            products,
            question=question,
        )
        if not last_errors:
            return answer
        if attempt < validation_attempts:
            if correction_delay:
                time.sleep(correction_delay)
            allowed_lines = allowed_community_evidence_lines(products)
            prompt = (
                f"{base_prompt}\n\n"
                "PREVIOUS_ANSWER_REJECTED:\n"
                f"{answer}\n\n"
                "VALIDATION_ERRORS:\n- "
                + "\n- ".join(last_errors)
                + "\n\nALLOWED_EVIDENCE_LINES:\n- "
                + ("\n- ".join(allowed_lines) if allowed_lines else "нет")
                + "\n\nПерепиши ответ полностью. Числовые evidence-фразы разрешено "
                "только дословно копировать из ALLOWED_EVIDENCE_LINES."
            )

    raise GeminiGenerationError(
        "ответ отклонён grounding-валидатором: " + "; ".join(last_errors)
    )
