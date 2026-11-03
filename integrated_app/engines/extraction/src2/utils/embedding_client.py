"""Dedicated Qwen text-embedding API client."""

from __future__ import annotations

import time
from http import HTTPStatus
from typing import Any, Dict, List, Optional

import numpy as np

from .config_loader import load_embedding_config


class ApiEmbeddingClient:
    """DashScope embedding client with separate document/query modes."""

    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.provider = str(config.get("provider") or "qwen").strip().lower()
        self.api_key = str(config.get("api_key") or "").strip()
        self.model = str(config.get("model") or "qwen3.7-text-embedding").strip()
        self.base_url = str(
            config.get("base_url") or "https://dashscope.aliyuncs.com/api/v1"
        ).rstrip("/")
        self.dimension = int(config.get("dimension") or 1024)
        self.batch_size = max(1, min(int(config.get("batch_size") or 10), 20))
        self.timeout_seconds = int(config.get("timeout_seconds") or 120)
        self.retry_count = int(config.get("retry_count") or 2)
        self.output_type = str(config.get("output_type") or "dense")
        self.query_instruction = str(config.get("query_instruction") or "").strip()
        self.last_usage: Dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

    @property
    def backend_name(self) -> str:
        return f"{self.provider}_api/{self.model}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.last_usage = {}
        return self._embed_batches(texts, text_type="document")

    def embed_query(self, text: str) -> List[float]:
        self.last_usage = {}
        return self._request([text], text_type="query", instruct=self.query_instruction)[0]

    def _embed_batches(self, texts: List[str], text_type: str) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._request(texts[start : start + self.batch_size], text_type=text_type))
        return vectors

    def _request(
        self,
        texts: List[str],
        text_type: str,
        instruct: Optional[str] = None,
    ) -> List[List[float]]:
        if not self.available:
            raise RuntimeError("Embedding API key/model/base URL is incomplete.")
        if not texts:
            return []

        try:
            import dashscope
        except ImportError as exc:
            raise RuntimeError("dashscope is required for the embedding API.") from exc

        dashscope.base_http_api_url = self.base_url
        kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "model": self.model,
            "input": texts,
            "dimension": self.dimension,
            "text_type": text_type,
            "output_type": self.output_type,
            "timeout": self.timeout_seconds,
        }
        if instruct and text_type == "query":
            kwargs["instruct"] = instruct

        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            try:
                response = dashscope.TextEmbedding.call(**kwargs)
                status_code = getattr(response, "status_code", None)
                if status_code not in {None, HTTPStatus.OK, 200}:
                    message = getattr(response, "message", "unknown error")
                    code = getattr(response, "code", status_code)
                    raise RuntimeError(f"DashScope embedding error {code}: {message}")
                output = getattr(response, "output", None) or {}
                items = list(output.get("embeddings") or [])
                if len(items) != len(texts):
                    raise RuntimeError(
                        f"Embedding count mismatch: requested {len(texts)}, received {len(items)}"
                    )
                items.sort(key=lambda item: int(item.get("text_index", 0)))
                vectors = [self._normalize(item.get("embedding") or []) for item in items]
                if any(len(vector) != self.dimension for vector in vectors):
                    actual = sorted({len(vector) for vector in vectors})
                    raise RuntimeError(
                        f"Embedding dimension mismatch: expected {self.dimension}, received {actual}"
                    )
                usage = dict(getattr(response, "usage", None) or {})
                for key, value in usage.items():
                    if isinstance(value, (int, float)):
                        self.last_usage[key] = self.last_usage.get(key, 0) + value
                    else:
                        self.last_usage[key] = value
                return vectors
            except Exception as exc:
                last_error = exc
                if attempt < self.retry_count:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Embedding API request failed: {last_error}")

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        vector = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()


def create_embedding_client(logger=None) -> Optional[ApiEmbeddingClient]:
    config = load_embedding_config()
    if str(config.get("enabled", True)).lower() in {"0", "false", "no", "off"}:
        return None
    client = ApiEmbeddingClient(config, logger=logger)
    if not client.available:
        raise RuntimeError("Embedding API is enabled but api_key/model/base_url is incomplete.")
    return client
