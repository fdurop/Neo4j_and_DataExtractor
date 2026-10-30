"""Single-request vision adapter for this run's shared, resumable budget."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src2"))
from utils.vlm_client import ApiVLMClient
from validation_budget import BudgetExceeded, BudgetStateError, _atomic_json


class BudgetedVisionClient(ApiVLMClient):
    def __init__(self, config, ledger, output, stage="fresh_extraction", max_stage_requests=65):
        super().__init__({**config, "retry_count": 0, "table_retry_count": 0, "enable_thinking": False})
        parsed = urlparse(self.base_url)
        if self.provider != "qwen" or self.model != ledger.settings["model"] or parsed.scheme != "https" or parsed.hostname != "dashscope.aliyuncs.com" or parsed.path != "/compatible-mode/v1/chat/completions" or parsed.port not in {None, 443} or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Only the existing Qwen/DashScope model and endpoint are permitted")
        if not self.api_key or not 1 <= max_stage_requests <= 100:
            raise ValueError("Missing key or invalid stage cap")
        self.ledger, self.stage = ledger, stage
        self.max_stage_requests = max_stage_requests
        self.path = Path(output).resolve() / "api_calls.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        if not isinstance(self.records, list) or any(r.get("budget_ledger_path") != str(ledger.path) or r.get("stage") != stage or r.get("requested_model") != self.model for r in self.records):
            raise BudgetStateError("Vision response log belongs to a different run, stage or model")
        ledger_ids = {r["id"] for r in ledger._load()["records"] if r["stage"] == stage}
        if any(r.get("budget_record_id") not in ledger_ids for r in self.records):
            raise BudgetStateError("Vision log is not backed by this shared ledger")
        self.cache = {r["cache_key"]: r for r in self.records if r.get("status") == "success"}
        self.cache_hits = 0
        self.last_fatal_error = None

    def _save(self):
        clean = json.loads(json.dumps(self.records, ensure_ascii=False).replace(self.api_key, "[REDACTED]"))
        _atomic_json(self.path, clean)

    def _chat_with_image(self, image_path, prompt, max_tokens=None, timeout_seconds=None, retry_count=None, operation="vision"):
        self.last_usage, self.last_response_metadata = {}, {}
        self.last_fatal_error = None
        image = Path(image_path).resolve()
        image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        output_limit = max_tokens or self.max_tokens
        cache_key = hashlib.sha256(json.dumps([self.model, image_hash, prompt, output_limit], ensure_ascii=False).encode("utf-8")).hexdigest()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            self.last_usage = copy.deepcopy(cached.get("usage") or {})
            self.last_response_metadata = copy.deepcopy(cached.get("response_metadata") or {})
            self.last_response_metadata.update(cache_reused_within_run=True, budget_record_id=cached["budget_record_id"])
            self.cache_hits += 1
            return cached["returned_text"]
        stage_attempts = max(len(self.records), self.ledger.snapshot()["stage_request_counts"].get(self.stage, 0))
        if stage_attempts >= self.max_stage_requests:
            raise BudgetExceeded("Vision phase request cap reached; no request sent")
        # Qwen3.7 has a documented per-image visual maximum of 16,384 tokens.
        # Use 20,000 plus a byte upper bound and framing allowance; never count
        # base64 bytes as literal text tokens or rely on uncertain image detail.
        reservation = self.ledger.reserve(20_000 + len(prompt.encode("utf-8")) + 1024, output_limit,
            stage=self.stage, request_ref=str(self.path), metadata={"operation": operation,
            "image_sha256": image_hash, "prompt_sha256": prompt_hash, "model": self.model})
        record = {"attempt": len(self.records) + 1, "stage": self.stage, "status": "started",
            "image": str(image), "image_sha256": image_hash, "prompt": prompt,
            "prompt_sha256": prompt_hash, "operation": operation, "requested_model": self.model,
            "cache_key": cache_key, "budget_record_id": reservation, "budget_ledger_path": str(self.ledger.path)}
        self.records.append(record)
        self._save()
        started, usage = time.perf_counter(), None
        try:
            payload = {"model": self.model, "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self._image_to_data_url(str(image)), "detail": self.image_detail}}]}],
                "temperature": self.temperature, "max_tokens": output_limit, "enable_thinking": False}
            response = requests.post(self.base_url, headers={"Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json"}, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=timeout_seconds or self.timeout_seconds, allow_redirects=False)
            record["http_status"] = response.status_code
            if response.status_code >= 300:
                raise RuntimeError(f"HTTP {response.status_code}; vision run stopped without retry")
            body = response.json()
            usage = body.get("usage")
            self.last_usage = copy.deepcopy(usage) if isinstance(usage, dict) else {}
            choice = body["choices"][0]
            self.last_response_metadata = {"response_id": body.get("id"), "model": body.get("model"),
                "finish_reason": choice.get("finish_reason"), "budget_record_id": reservation}
            record.update(usage=self.last_usage, response_metadata=self.last_response_metadata)
            text = choice["message"]["content"]
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
            record["returned_text"] = text if isinstance(text, str) else ""
            if choice.get("finish_reason") != "stop":
                raise ValueError("Vision answer did not finish normally; retained for review, not a successful extraction")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Vision answer has no visible content")
            record["status"] = "success"
            self.cache[cache_key] = record
            return text.strip()
        except Exception as error:
            safe_error = str(error).replace(self.api_key, "[REDACTED]")
            record.update(status="error", error=safe_error)
            self.last_fatal_error = safe_error
            raise RuntimeError(safe_error) from None
        finally:
            record["seconds"] = round(time.perf_counter() - started, 3)
            self.ledger.complete(reservation, usage, status="success" if record["status"] == "success" else "error",
                metadata={"http_status": record.get("http_status"), "finish_reason": self.last_response_metadata.get("finish_reason"),
                "seconds": record["seconds"], "result_status": record["status"]})
            self._save()
            print(f"{self.stage} request {record['attempt']}/{self.max_stage_requests}: {record['status']} ({record['seconds']}s)", flush=True)
