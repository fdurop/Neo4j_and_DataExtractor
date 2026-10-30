"""Fail-closed, shared cost/request ledger for the explicitly bounded Qwen test.

Amounts use integer micro-yuan. This is a conservative estimate at undiscounted
Qwen3.7 Plus Beijing prices, not a query of the provider's account or bill.
Callers must reserve a conservative input-token upper bound BEFORE sending.
Missing/invalid usage and interrupted calls retain their entire reservation.
No retries are performed. Existing ledger limits cannot be raised on resume.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid


class BudgetExceeded(RuntimeError):
    """No further HTTP request may be made under this ledger."""


class BudgetStateError(RuntimeError):
    """Ledger is inconsistent, locked, or configured differently on resume."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _money(micro_yuan):
    return format(Decimal(micro_yuan) / Decimal(1_000_000), ".6f")


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


_META_KEYS = {
    "operation", "prompt_sha256", "image_sha256", "log_path", "http_status",
    "finish_reason", "response_id", "model", "error_type", "result_status",
    "seconds", "page_no", "image_index", "source_kind", "task_id", "note",
}


def _metadata(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - _META_KEYS:
        raise ValueError("Ledger metadata must use documented safe scalar keys")
    result = {}
    for key, item in value.items():
        if item is None or isinstance(item, (bool, int)):
            result[key] = item
        elif isinstance(item, float) and item == item and abs(item) < 1e12:
            result[key] = item
        elif isinstance(item, str) and len(item) <= 600 and not item.startswith("data:"):
            result[key] = item
        else:
            raise ValueError("Ledger metadata cannot contain payloads or unbounded values")
    return result


class ValidationBudget:
    """Cross-process-resumable estimate capped at 10 yuan and 100 attempts.

    A short exclusive lock file makes concurrent updates fail closed. A lock
    left by a killed process must be inspected rather than automatically erased.
    The module cannot establish whether a caller-supplied input bound is valid.
    """

    MAX_INPUT_TOKENS = 256_000  # deliberately below 256 * 1024 tier boundary
    MAX_OUTPUT_TOKENS = 8192
    MODELS = {"qwen3.7-plus", "qwen3.7-plus-2026-05-26"}

    def __init__(self, path, max_cost_yuan="10", max_requests=100, model="qwen3.7-plus"):
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        _integer(max_requests, "max_requests", 1)
        if max_requests > 100 or model not in self.MODELS:
            raise ValueError("This test permits at most 100 attempts and the specified Qwen3.7 model")
        try:
            amount = Decimal(str(max_cost_yuan))
        except InvalidOperation:
            raise ValueError("Invalid budget amount") from None
        if not amount.is_finite() or not 0 < amount <= 10 or amount * 1_000_000 != (amount * 1_000_000).to_integral_value():
            raise ValueError("Budget must be positive, <= 10 yuan, with <= 6 decimal places")
        self.settings = {
            "schema_version": 1, "model": model,
            "max_cost_micro_yuan": int(amount * 1_000_000), "max_requests": max_requests,
            "input_micro_yuan_per_token": 2, "output_micro_yuan_per_token": 8,
            "max_input_tokens_per_request": self.MAX_INPUT_TOKENS,
            "pricing_basis": "Qwen3.7 Plus Beijing undiscounted <=256K input; no cache/free/discount deduction",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if self.path.exists():
                self._load()
            else:
                _atomic_json(self.path, dict(self.settings, created_at=_now(), records=[]))

    @contextmanager
    def _locked(self):
        try:
            handle = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise BudgetStateError("Budget ledger is locked; inspect its owner before resuming") from None
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(json.dumps({"pid": os.getpid(), "created_at": _now()}))
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)

    def _load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if any(data.get(key) != expected for key, expected in self.settings.items()):
                raise ValueError("configuration differs")
            records = data["records"]
            if not isinstance(records, list) or len(records) > self.settings["max_requests"]:
                raise ValueError("invalid records")
            ids = set()
            for record in records:
                if not isinstance(record, dict) or record["id"] in ids:
                    raise ValueError("invalid record ID")
                ids.add(record["id"])
                _integer(record["reserved_micro_yuan"], "reserved_micro_yuan")
                _integer(record["accounted_micro_yuan"], "accounted_micro_yuan")
                input_bound = _integer(record["input_upper_bound"], "input_upper_bound", 1)
                output_bound = _integer(record["max_output_tokens"], "max_output_tokens", 1)
                if (input_bound > self.MAX_INPUT_TOKENS or output_bound > self.MAX_OUTPUT_TOKENS
                        or record["reserved_micro_yuan"] != input_bound * 2 + output_bound * 8):
                    raise ValueError("invalid reservation")
                if not isinstance(record["stage"], str) or not record["stage"]:
                    raise ValueError("invalid stage")
                if record["accounting"] not in {"reserved", "reserved_unknown", "measured"}:
                    raise ValueError("invalid accounting state")
                if record["accounting"] != "measured" and record["accounted_micro_yuan"] != record["reserved_micro_yuan"]:
                    raise ValueError("an unresolved request lost its reservation")
                if record["accounting"] == "measured":
                    usage = record["usage"]
                    prompt = _integer(usage["prompt_tokens"], "prompt_tokens")
                    completion = _integer(usage["completion_tokens"], "completion_tokens")
                    if usage["total_tokens"] != prompt + completion:
                        raise ValueError("invalid measured usage")
                    expected = (3 if prompt > self.MAX_INPUT_TOKENS else 1) * (prompt * 2 + completion * 8)
                    if record["accounted_micro_yuan"] != expected:
                        raise ValueError("invalid measured cost")
                    if record.get("bound_violation") != (prompt > input_bound or completion > output_bound):
                        raise ValueError("invalid bound status")
            return data
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise BudgetStateError(f"Budget ledger invalid or incompatible ({type(error).__name__})") from None

    @staticmethod
    def _totals(data):
        records = data["records"]
        measured = sum(r["accounted_micro_yuan"] for r in records if r["accounting"] == "measured")
        unresolved = sum(r["accounted_micro_yuan"] for r in records if r["accounting"] != "measured")
        committed = measured + unresolved
        return {
            "request_count": len(records), "max_requests": data["max_requests"],
            "remaining_requests": max(0, data["max_requests"] - len(records)),
            "estimated_cost_yuan": _money(measured), "reserved_unknown_yuan": _money(unresolved),
            "committed_cost_yuan": _money(committed),
            "remaining_yuan": _money(max(0, data["max_cost_micro_yuan"] - committed)),
            "max_cost_yuan": _money(data["max_cost_micro_yuan"]),
            "unresolved_count": sum(r["accounting"] != "measured" for r in records),
            "blocked_due_to_bound_violation": any(r.get("bound_violation") for r in records),
            "account_balance_queried": False,
            "stage_request_counts": {stage: sum(r["stage"] == stage for r in records)
                                     for stage in sorted({r["stage"] for r in records})},
        }

    def snapshot(self):
        return self._totals(self._load())

    def reserve(self, input_upper_bound, max_output_tokens, *, stage, request_ref="", metadata=None):
        _integer(input_upper_bound, "input_upper_bound", 1)
        _integer(max_output_tokens, "max_output_tokens", 1)
        if input_upper_bound > self.MAX_INPUT_TOKENS or max_output_tokens > self.MAX_OUTPUT_TOKENS:
            raise BudgetExceeded("Request exceeds this test's token bounds; no request reserved")
        if not isinstance(stage, str) or not stage or len(stage) > 100:
            raise ValueError("stage must be a short nonempty string")
        if not isinstance(request_ref, str) or len(request_ref) > 600 or request_ref.startswith("data:"):
            raise ValueError("request_ref must be a short non-payload reference")
        safe_metadata = _metadata(metadata)
        reserve_cost = input_upper_bound * 2 + max_output_tokens * 8
        with self._locked():
            data = self._load()
            total = self._totals(data)
            if total["blocked_due_to_bound_violation"]:
                raise BudgetExceeded("An earlier request exceeded its token bound; inspect before resuming")
            if total["remaining_requests"] <= 0:
                raise BudgetExceeded("Shared request limit exhausted; no HTTP request permitted")
            committed = sum(r["accounted_micro_yuan"] for r in data["records"])
            if committed + reserve_cost > data["max_cost_micro_yuan"]:
                raise BudgetExceeded("Shared cost estimate plus this reservation exceeds budget; no HTTP request permitted")
            record_id = uuid.uuid4().hex
            data["records"].append({
                "id": record_id, "attempt": len(data["records"]) + 1,
                "stage": stage, "request_ref": request_ref, "started_at": _now(),
                "status": "started", "input_upper_bound": input_upper_bound,
                "max_output_tokens": max_output_tokens, "reserved_micro_yuan": reserve_cost,
                "accounted_micro_yuan": reserve_cost, "accounting": "reserved", "metadata": safe_metadata,
            })
            _atomic_json(self.path, data)
        return record_id

    def complete(self, record_id, usage=None, *, status="success", metadata=None):
        if status not in {"success", "error", "invalid_response", "interrupted"}:
            raise ValueError("Unsupported completion status")
        safe_metadata = _metadata(metadata)
        with self._locked():
            data = self._load()
            matches = [r for r in data["records"] if r["id"] == record_id]
            if len(matches) != 1:
                raise BudgetStateError("Unknown reserved request")
            record = matches[0]
            if record["status"] != "started":
                raise BudgetStateError("Request already finalized; its accounting cannot be silently changed")
            valid = False
            if isinstance(usage, dict):
                try:
                    prompt = _integer(usage.get("prompt_tokens"), "prompt_tokens")
                    completion = _integer(usage.get("completion_tokens"), "completion_tokens")
                    total = usage.get("total_tokens", prompt + completion)
                    _integer(total, "total_tokens")
                    valid = total == prompt + completion
                except ValueError:
                    valid = False
            record.update(status=status, finished_at=_now())
            record["metadata"].update(safe_metadata)
            if valid:
                record["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
                # Higher-tier defensive accounting if a caller's bound was wrong.
                multiplier = 3 if prompt > self.MAX_INPUT_TOKENS else 1
                record["accounted_micro_yuan"] = multiplier * (prompt * 2 + completion * 8)
                record["accounting"] = "measured"
                record["bound_violation"] = prompt > record["input_upper_bound"] or completion > record["max_output_tokens"]
            else:
                record["usage"] = None
                record["accounting"] = "reserved_unknown"
                record["usage_note"] = "Missing, malformed or inconsistent usage; full reservation retained"
            _atomic_json(self.path, data)
            return dict(record)


class BudgetedQwenJSONClient:
    """One-shot Qwen JSON calls, resumable local response cache and shared ledger.

    Full prompts/responses belong only to the phase api_results.json; the shared
    ledger receives hashes/references and token usage. Config/secrets never log.
    """

    def __init__(self, config, ledger, output, stage, max_stage_requests=35, max_output_tokens=5000):
        self.config = dict(config)
        self.model = str(config.get("model") or "")
        self.url = str(config.get("base_url") or "")
        parsed = urlparse(self.url)
        if (str(config.get("provider", "qwen")).lower() != "qwen"
                or parsed.scheme != "https" or parsed.hostname != "dashscope.aliyuncs.com"
                or parsed.path.rstrip("/") != "/compatible-mode/v1/chat/completions"
                or parsed.port not in {None, 443} or parsed.username or parsed.password or parsed.query or parsed.fragment
                or not config.get("api_key")):
            raise ValueError("Expected the configured Qwen DashScope chat endpoint and credentials")
        if self.model != ledger.settings["model"]:
            raise ValueError("Client model differs from shared ledger model")
        _integer(max_stage_requests, "max_stage_requests", 1)
        if max_stage_requests > 100:
            raise ValueError("Invalid stage request cap")
        _integer(max_output_tokens, "max_output_tokens", 1)
        if max_output_tokens > ValidationBudget.MAX_OUTPUT_TOKENS:
            raise ValueError("max_output_tokens must be within 1..8192")
        self.ledger = ledger
        self.stage = stage
        self.max_stage_requests = max_stage_requests
        self.max_output_tokens = max_output_tokens
        self.path = Path(output) / "api_results.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        ledger_data = self.ledger._load()
        ledger_ids = {r["id"] for r in ledger_data["records"] if r["stage"] == stage}
        if not isinstance(self.records, list) or any(
                r.get("stage") != stage or r.get("model") != self.model
                or r.get("budget_ledger_path") != str(self.ledger.path)
                or r.get("budget_record_id") not in ledger_ids for r in self.records):
            raise BudgetStateError("Output log belongs to another stage or model")
        self.cache = {r["cache_key"]: r["response"] for r in self.records
                      if r.get("status") == "success" and isinstance(r.get("response"), dict)}

    @property
    def remaining_requests(self):
        snapshot = self.ledger.snapshot()
        stage_attempts = max(len(self.records), snapshot["stage_request_counts"].get(self.stage, 0))
        return max(0, min(self.max_stage_requests - stage_attempts, snapshot["remaining_requests"]))

    def stage_usage(self):
        return {key: sum(r.get("usage", {}).get(key, 0) for r in self.records if isinstance(r.get("usage"), dict))
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")}

    def _redact(self, value):
        encoded = json.dumps(value, ensure_ascii=False)
        return json.loads(encoded.replace(str(self.config["api_key"]), "[REDACTED]"))

    def _save(self):
        _atomic_json(self.path, self._redact(self.records))

    def chat_json(self, prompt, **unused):
        import requests
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be nonempty text")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_key = hashlib.sha256((self.model + "\0" + prompt).encode("utf-8")).hexdigest()
        if cache_key in self.cache:
            return copy.deepcopy(self.cache[cache_key])
        if self.remaining_requests <= 0:
            raise BudgetExceeded("Phase or shared HTTP request cap exhausted")
        reservation = self.ledger.reserve(len(prompt.encode("utf-8")) + 512, self.max_output_tokens,
            stage=self.stage, request_ref=str(self.path),
            metadata={"operation": "graph_json", "prompt_sha256": digest, "model": self.model})
        record = {"request": len(self.records) + 1, "stage": self.stage, "model": self.model,
                  "provider": "qwen", "status": "started", "prompt": prompt,
                  "prompt_sha256": digest, "cache_key": cache_key,
                  "max_output_tokens": self.max_output_tokens,
                  "budget_record_id": reservation, "budget_ledger_path": str(self.ledger.path),
                  "started_at": _now()}
        self.records.append(record)
        self._save()
        started = time.perf_counter()
        usage = None
        try:
            response = requests.post(self.url,
                headers={"Authorization": "Bearer " + str(self.config["api_key"])},
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.05, "max_tokens": self.max_output_tokens, "enable_thinking": False,
                      "response_format": {"type": "json_object"}},
                timeout=self.config.get("timeout_seconds", 120), allow_redirects=False)
            record["http_status"] = response.status_code
            if response.status_code >= 300:
                raise RuntimeError(f"HTTP {response.status_code}")
            body = response.json()
            usage = body.get("usage")
            record["usage"] = usage if isinstance(usage, dict) else {}
            record["raw_response"] = self._redact(body)
            choice = body["choices"][0]
            record["finish_reason"] = choice.get("finish_reason")
            if choice.get("finish_reason") in {"length", "content_filter"}:
                raise ValueError("Output truncated or filtered")
            result = json.loads(choice["message"]["content"])
            if not isinstance(result, dict):
                raise ValueError("Response must be a JSON object")
            record.update(status="success", response=result)
            self.cache[cache_key] = copy.deepcopy(result)
            return copy.deepcopy(result)
        except Exception as error:
            record.update(status="error", error=str(error).replace(str(self.config["api_key"]), "[REDACTED]"))
            raise RuntimeError(record["error"]) from None
        finally:
            record["seconds"] = round(time.perf_counter() - started, 3)
            self.ledger.complete(reservation, usage, status="success" if record["status"] == "success" else "error",
                metadata={"http_status": record.get("http_status"), "finish_reason": record.get("finish_reason"),
                          "seconds": record["seconds"], "result_status": record["status"]})
            self._save()
            print(f"{self.stage} request {record['request']}/{self.max_stage_requests}: {record['status']} ({record['seconds']}s)", flush=True)
