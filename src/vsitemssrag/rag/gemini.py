"""Минимальный клиент Gemini Generate Content API для RAG-ответов."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from vsitemssrag.embeddings.gemini import (
    GEMINI_API_BASE_URL,
    RETRYABLE_STATUS_CODES,
    extract_error_message,
)


GENERATION_MODEL = "gemini-3.5-flash-lite"


class GeminiGenerationError(RuntimeError):
    """Gemini не смог вернуть пригодный текстовый ответ."""


def parse_generation_response(payload: Mapping[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        feedback = payload.get("promptFeedback")
        reason = feedback.get("blockReason") if isinstance(feedback, Mapping) else None
        suffix = f": {reason}" if reason else ""
        raise GeminiGenerationError(f"Gemini не вернул кандидата ответа{suffix}")

    candidate = candidates[0]
    content = candidate.get("content") if isinstance(candidate, Mapping) else None
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list):
        raise GeminiGenerationError("Gemini вернул ответ без content.parts")

    text_parts = [
        str(part["text"]).strip()
        for part in parts
        if isinstance(part, Mapping) and str(part.get("text") or "").strip()
    ]
    if not text_parts:
        finish_reason = (
            candidate.get("finishReason") if isinstance(candidate, Mapping) else None
        )
        suffix = f": {finish_reason}" if finish_reason else ""
        raise GeminiGenerationError(f"Gemini вернул пустой текст{suffix}")
    return "\n".join(text_parts)


class GeminiGenerationClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = GENERATION_MODEL,
        timeout: float = 60.0,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key не может быть пустым")
        if not model.strip() or timeout <= 0:
            raise ValueError("некорректные параметры GeminiGenerationClient")

        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = session or requests.Session()
        self._owns_session = session is None

        if self._owns_session:
            retries = Retry(
                total=3,
                connect=3,
                read=3,
                status=3,
                backoff_factor=1,
                status_forcelist=RETRYABLE_STATUS_CODES,
                allowed_methods=frozenset({"POST"}),
                respect_retry_after_header=True,
                raise_on_status=False,
            )
            self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "GeminiGenerationClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str,
        max_output_tokens: int = 1200,
        temperature: float | None = None,
    ) -> str:
        if not prompt.strip() or not system_instruction.strip():
            raise ValueError("prompt и system_instruction не могут быть пустыми")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens должен быть больше нуля")
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError("temperature должна быть от 0 до 2")

        generation_config: dict[str, Any] = {
            "maxOutputTokens": max_output_tokens,
        }
        if temperature is not None:
            generation_config["temperature"] = temperature

        response = self.session.post(
            f"{GEMINI_API_BASE_URL}/{self.model}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            json={
                "systemInstruction": {
                    "parts": [{"text": system_instruction}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": generation_config,
            },
            timeout=(10, self.timeout),
        )
        if not response.ok:
            raise GeminiGenerationError(
                f"Gemini API HTTP {response.status_code}: "
                f"{extract_error_message(response)}"
            )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise GeminiGenerationError("Gemini вернул некорректный JSON")
        return parse_generation_response(payload)
