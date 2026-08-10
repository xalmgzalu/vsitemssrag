"""Минимальный клиент Gemini Embeddings API."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


class GeminiEmbeddingError(RuntimeError):
    """Gemini вернул некорректный ответ или отказал в запросе."""


def parse_embedding_response(
    payload: Mapping[str, Any], dimensions: int = EMBEDDING_DIMENSIONS
) -> list[float]:
    embedding = payload.get("embedding")
    if not isinstance(embedding, Mapping):
        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            embedding = embeddings[0]

    if not isinstance(embedding, Mapping):
        raise GeminiEmbeddingError("Gemini не вернул embedding")

    raw_values = embedding.get("values")
    if not isinstance(raw_values, list):
        raise GeminiEmbeddingError("Gemini вернул embedding без values")

    try:
        values = [float(value) for value in raw_values]
    except (TypeError, ValueError) as error:
        raise GeminiEmbeddingError("Gemini вернул нечисловой embedding") from error

    if len(values) != dimensions:
        raise GeminiEmbeddingError(
            f"ожидался embedding размерности {dimensions}, получено {len(values)}"
        )
    if not all(math.isfinite(value) for value in values):
        raise GeminiEmbeddingError("Gemini вернул NaN или Infinity")
    return values


def extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.reason or "неизвестная ошибка"

    error = payload.get("error") if isinstance(payload, Mapping) else None
    if isinstance(error, Mapping) and error.get("message"):
        return str(error["message"])
    return response.reason or "неизвестная ошибка"


class GeminiEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIMENSIONS,
        delay: float = 1.0,
        timeout: float = 45.0,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key не может быть пустым")
        if dimensions <= 0 or delay < 0 or timeout <= 0:
            raise ValueError("некорректные параметры GeminiEmbeddingClient")

        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.delay = delay
        self.timeout = timeout
        self.last_request_finished_at: float | None = None
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
            )
            self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "GeminiEmbeddingClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("текст для embedding не может быть пустым")

        if self.last_request_finished_at is not None:
            elapsed = time.monotonic() - self.last_request_finished_at
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)

        try:
            response = self.session.post(
                f"{GEMINI_API_BASE_URL}/{self.model}:embedContent",
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                json={
                    "content": {"parts": [{"text": text}]},
                    "output_dimensionality": self.dimensions,
                },
                timeout=(10, self.timeout),
            )
        finally:
            self.last_request_finished_at = time.monotonic()

        if not response.ok:
            raise GeminiEmbeddingError(
                f"Gemini API HTTP {response.status_code}: "
                f"{extract_error_message(response)}"
            )
        return parse_embedding_response(response.json(), self.dimensions)
