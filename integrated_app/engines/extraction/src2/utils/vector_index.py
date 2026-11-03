"""Vector index writer using a dedicated Qwen embedding API."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .config_loader import load_embedding_config
from .embedding_client import ApiEmbeddingClient, create_embedding_client
from .path_config import workspace_relative


def _relpath(path: str) -> str:
    return workspace_relative(path)


def _tokens(text: str) -> Iterable[str]:
    text = (text or "").strip().lower()
    if not text:
        return []

    ascii_words = re.findall(r"[a-z0-9_+\-./]+", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    chinese_bigrams = [
        "".join(chinese_chars[i : i + 2]) for i in range(max(0, len(chinese_chars) - 1))
    ]
    return ascii_words + chinese_chars + chinese_bigrams


def hash_embedding(text: str, dim: int = 384) -> List[float]:
    vector = np.zeros(dim, dtype=np.float32)
    for token in _tokens(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector.tolist()


def build_vector_index(
    records: List[Dict[str, Any]],
    output_dir: str,
    dim: int = 384,
    backend: str = "hash_text_v1",
    embedding_client: Optional[ApiEmbeddingClient] = None,
    logger=None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    matrix_path = os.path.join(output_dir, "vector_matrix.npy")
    meta_path = os.path.join(output_dir, "vector_index.json")

    texts = [
        str(
            record.get("embedding_text")
            or record.get("text")
            or record.get("description")
            or record.get("name")
            or record.get("node_id")
            or "empty node"
        ).strip()
        for record in records
    ]

    try:
        client = embedding_client if embedding_client is not None else create_embedding_client(logger)
        if client is not None:
            backend = client.backend_name
            dim = client.dimension
            vectors = client.embed_documents(texts)
            model = client.model
            provider = client.provider
            usage = client.last_usage
        else:
            vectors = [hash_embedding(text, dim=dim) for text in texts]
            model = None
            provider = "local"
            usage = {}
        status = "success"
        error = None
    except Exception as exc:
        config = load_embedding_config()
        dim = int(config.get("dimension") or 0)
        backend = "embedding_api_failed"
        model = str(config.get("model") or "")
        provider = str(config.get("provider") or "qwen")
        usage = {}
        vectors = []
        status = "failed"
        error = str(exc)
        if logger:
            logger.error("Embedding API index generation failed: %s", exc, exc_info=True)

    metadata = []
    if status == "success":
        for record, embedding_text in zip(records, texts):
            embedding_id = record.get("embedding_id") or (
                f"emb_{record.get('node_id') or record.get('element_id')}"
            )
            metadata.append(
                {
                    "embedding_id": embedding_id,
                    "node_id": record.get("node_id") or record.get("element_id"),
                    "node_type": record.get("node_type") or record.get("type"),
                    "user_id": record.get("user_id"),
                    "username": record.get("username"),
                    "course_id": record.get("course_id"),
                    "course_name": record.get("course_name"),
                    "job_id": record.get("job_id"),
                    "document_id": record.get("document_id"),
                    "page_id": record.get("page_id") or record.get("source_page_id"),
                    "embedding_text": embedding_text,
                    "backend": backend,
                    "model": model,
                    "dim": dim,
                }
            )

    matrix = (
        np.asarray(vectors, dtype=np.float32)
        if vectors
        else np.zeros((0, dim), dtype=np.float32)
    )
    np.save(matrix_path, matrix)
    index_payload = {
        "status": status,
        "backend": backend,
        "provider": provider,
        "model": model,
        "dim": dim,
        "text_type": "document",
        "output_type": "dense",
        "embedding_text_policy": "cleaned_v1",
        "count": len(metadata),
        "matrix_file": os.path.basename(matrix_path),
        "usage": usage,
        "items": metadata,
    }
    if error:
        index_payload["error"] = error
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, ensure_ascii=False, indent=2)

    if logger and status == "success":
        logger.info(
            "Vector index completed: backend=%s model=%s dim=%s count=%s usage=%s",
            backend,
            model,
            dim,
            len(metadata),
            usage,
        )

    summary = {
        "status": status,
        "backend": backend,
        "model": model,
        "dim": dim,
        "count": len(metadata),
        "matrix_path": _relpath(matrix_path),
        "metadata_path": _relpath(meta_path),
    }
    if error:
        summary["error"] = error
    return summary


def cosine_search(query: str, index_dir: str, top_k: int = 5) -> List[Dict[str, Any]]:
    meta_path = os.path.join(index_dir, "vector_index.json")
    matrix_path = os.path.join(index_dir, "vector_matrix.npy")
    if not os.path.exists(meta_path) or not os.path.exists(matrix_path):
        return []

    with open(meta_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    if index.get("status") == "failed":
        raise RuntimeError(f"Vector index is unavailable: {index.get('error', 'unknown error')}")
    matrix = np.load(matrix_path)
    if matrix.size == 0:
        return []

    if index.get("backend") == "hash_text_v1":
        query_values = hash_embedding(query, dim=index.get("dim", 384))
    else:
        client = create_embedding_client()
        if client is None:
            raise RuntimeError("Semantic query requires the configured embedding API.")
        if client.model != index.get("model") or client.dimension != index.get("dim"):
            raise RuntimeError("Query embedding model/dimension does not match the stored index.")
        query_values = client.embed_query(query)
    query_vec = np.asarray(query_values, dtype=np.float32)
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:top_k]
    results = []
    for rank, idx in enumerate(order, 1):
        item = dict(index["items"][int(idx)])
        item["rank"] = rank
        item["score"] = float(scores[int(idx)])
        results.append(item)
    return results
