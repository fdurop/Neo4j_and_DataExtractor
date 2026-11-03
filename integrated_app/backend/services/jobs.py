"""Persistent document jobs. Completion includes publication; failures remain visible."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import copy
import re
from pathlib import Path
from threading import RLock
from urllib.parse import quote
import uuid

from werkzeug.exceptions import Conflict, NotFound
from backend.services.course_store import atomic_json, now, read_json

ACTIVE = {"queued", "running", "cancel_requested"}
TERMINAL = {"succeeded", "failed", "cancelled"}


class JobCancelled(RuntimeError):
    pass


class JobService:
    def __init__(self, app, directory):
        self.app, self.directory = app, Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.executor = ThreadPoolExecutor(max_workers=int(app.config["JOB_WORKERS"]), thread_name_prefix="course-job")
        self.futures = {}
        self._recover()

    def _path(self, identity):
        if not isinstance(identity, str) or not re.fullmatch(r"job_[a-f0-9]{32}", identity):
            raise NotFound("任务不存在。")
        return self.directory / identity / "job.json"

    def _read(self, identity):
        record = read_json(self._path(identity), None)
        if not isinstance(record, dict):
            raise NotFound("任务不存在。")
        return record

    def _save(self, record):
        record["updated_at"] = now()
        atomic_json(self._path(record["id"]), record)

    def _recover(self):
        with self.lock:
            for path in self.directory.glob("job_*/job.json"):
                record = read_json(path, None)
                if isinstance(record, dict) and record.get("status") in ACTIVE:
                    record.update(status="failed", stage="interrupted", error="服务重启，任务已中断。请重新上传；系统不会自动重复调用API。",
                                  error_code="interrupted", finished_at=now())
                    self._save(record)

    def create(self, course, original_filename, source, page_count, owner_id=None):
        identity = "job_" + uuid.uuid4().hex
        job_dir = self.directory / identity
        job_dir.mkdir()
        target = job_dir / "source.pdf"
        # The staged, validated source lives under the same data root.
        Path(source).replace(target)
        import hashlib
        record = {"id": identity, "job_id": identity, "course_id": course["id"], "course_name": course["name"],
                  "original_filename": original_filename, "filename": original_filename, "owner_id": owner_id,
                  "input_path": str(target), "job_dir": str(job_dir), "size": target.stat().st_size,
                  "source_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "page_count": page_count,
                  "created_at": now(), "status": "queued", "stage": "queued", "progress": 0,
                  "message": "等待处理", "error": None, "cancel_requested": False, "artifacts": {},
                  "source_url": "/api/jobs/" + identity + "/source"}
        with self.lock:
            self._save(record)
        if self.app.config["AUTO_START_JOBS"]:
            self.start(identity)
        return self.get(identity)

    def start(self, identity):
        with self.lock:
            record = self._read(identity)
            if record["status"] != "queued" or identity in self.futures:
                raise Conflict("该任务不能重复启动。")
            self.futures[identity] = self.executor.submit(self._run, identity)

    def retry(self, identity):
        with self.lock:
            record = self._read(identity)
            if record["status"] not in {"failed", "cancelled"}:
                raise Conflict("只有失败或已取消任务可以恢复。")
            old_future = self.futures.get(identity)
            if old_future is not None and not old_future.done():
                raise Conflict("原处理进程尚未结束。")
            self.futures.pop(identity, None)
            record.update(status="queued", stage="queued", error=None, resume=True,
                          cancel_requested=False, message="恢复任务，复用本次已完成的结果")
            record.pop("finished_at", None)
            self._save(record)
        self.start(identity)
        return self.get(identity)

    def get(self, identity, *, private=False):
        with self.lock:
            record = copy.deepcopy(self._read(identity))
        if private:
            return record
        record.pop("input_path", None)
        record.pop("job_dir", None)
        return record

    def list(self, course_id=None, *, include_hidden=False):
        with self.lock:
            identities = [path.parent.name for path in self.directory.glob("job_*/job.json")]
            records = [self.get(identity) for identity in identities]
        return sorted([item for item in records if (not course_id or item["course_id"] == course_id)
                       and (include_hidden or not item.get("hidden"))], key=lambda item: item["created_at"], reverse=True)

    def cancel(self, identity):
        with self.lock:
            record = self._read(identity)
            if record["status"] in TERMINAL:
                if record["status"] == "cancelled":
                    return self.get(identity)
                raise Conflict("任务已结束，无法取消。")
            record["cancel_requested"] = True
            if record["status"] == "queued":
                record.update(status="cancelled", stage="cancelled", finished_at=now(), message="任务已取消，未开始处理。")
                future = self.futures.get(identity)
                if future:
                    future.cancel()
            else:
                record.update(status="cancel_requested", message="已请求取消，将在安全阶段边界停止。")
            self._save(record)
        return self.get(identity)

    def hide(self, identity):
        with self.lock:
            record = self._read(identity)
            if record["status"] in ACTIVE:
                raise Conflict("任务仍在处理中，请先取消或等待结束。")
            record.update(hidden=True, hidden_at=now())
            self._save(record)

    def _safe_error(self, error):
        message = str(getattr(error, "public_message", None) or error)[:1000]
        for key, value in self.app.config.items():
            if any(word in key.upper() for word in ("PASSWORD", "SECRET", "API_KEY", "TOKEN")):
                if isinstance(value, str) and len(value) > 5:
                    message = message.replace(value, "[redacted]")
        message = re.sub(r"(?i)(?:sk-|pcsk_)[A-Za-z0-9_-]{10,}", "[redacted]", message)
        message = re.sub(r"(?i)((?:api[_-]?key|password|authorization|token)\s*[:=]\s*)[^\s,;]+", r"\1[redacted]", message)
        return message or "任务处理失败，请检查运行配置。"

    def _publish_local(self, job, result):
        graph = result.get("graph")
        if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("links"), list):
            raise ValueError("流水线未返回有效图谱，任务未发布。")
        from backend.routes import knowledge_graph
        with knowledge_graph._write_lock:
            graphs = knowledge_graph.load_graphs()
            graphs[job["course_name"]] = graph
            knowledge_graph.save_graphs(graphs)
        return {"graph_url": "/knowledge-graph/?course_name=" + quote(job["course_name"]) + "&course_id=" + job["course_id"]}

    def _run(self, identity):
        with self.app.app_context():
            with self.lock:
                job = self._read(identity)
                if job["status"] == "cancelled":
                    return
                job.update(status="running", stage="starting", started_at=now(), message="开始处理")
                self._save(job)

            def cancelled():
                with self.lock:
                    return bool(self._read(identity).get("cancel_requested"))

            def update(stage=None, progress=None, message=None, **fields):
                with self.lock:
                    current = self._read(identity)
                    if stage is not None:
                        current["stage"] = str(stage)
                    if progress is not None:
                        current["progress"] = max(current["progress"], min(99, max(0, int(progress))))
                    if message is not None:
                        current["message"] = str(message)
                    # Only a small set of public metadata is accepted from a worker.
                    for key in ("artifacts", "database", "api_budget", "warnings"):
                        if key in fields:
                            current[key] = fields[key]
                    self._save(current)

            try:
                runner = self.app.config.get("PIPELINE_RUNNER")
                if runner is None:
                    from backend.services.pipeline import run_job
                    runner = run_job
                result = runner(copy.deepcopy(job), update, cancelled)
                if cancelled():
                    raise JobCancelled("任务已在阶段边界停止，中间产物保留。")
                if not isinstance(result, dict):
                    raise ValueError("流水线没有返回任务结果。")
                update(stage="publishing", progress=97, message="保存课程图谱")
                publisher = self.app.config.get("PIPELINE_PUBLISHER") or self._publish_local
                publication = publisher(copy.deepcopy(job), result) or {}
                with self.lock:
                    current = self._read(identity)
                    late_cancel = current.get("cancel_requested", False)
                    current.update(status="succeeded", stage="completed", progress=100, error=None, finished_at=now(),
                                   message="处理完成" if not late_cancel else "处理已完成，取消请求到达时已进入最终保存阶段。",
                                   artifacts=result.get("artifacts", {}), database=result.get("database"),
                                   graph_url=publication.get("graph_url") or "/knowledge-graph/?course_name=" + quote(job["course_name"]) + "&course_id=" + job["course_id"])
                    for key in ("database", "warnings", "api_budget"):
                        if key in publication:
                            current[key] = publication[key]
                    self._save(current)
            except JobCancelled as error:
                with self.lock:
                    current = self._read(identity)
                    current.update(status="cancelled", stage="cancelled", message=str(error), finished_at=now())
                    self._save(current)
            except Exception as error:
                with self.lock:
                    current = self._read(identity)
                    # A requested cancellation only becomes cancelled when the
                    # worker confirms cancellation, not when another failure occurs.
                    current.update(status="failed", error=self._safe_error(error), error_type=type(error).__name__,
                                   message="任务失败，详情已保留。", finished_at=now())
                    self._save(current)
                self.app.logger.error("Job %s failed (%s)", identity, type(error).__name__)

    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait, cancel_futures=True)
