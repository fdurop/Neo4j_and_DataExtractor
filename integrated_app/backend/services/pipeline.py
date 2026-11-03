"""Run each paid extraction in its own process and retain durable artifacts."""
import json
from pathlib import Path
import subprocess
import sys
import time
from flask import current_app

ROOT = Path(__file__).resolve().parents[2]


def run_job(job, update, cancelled):
    config = dict(current_app.config) if current_app else {}
    if not config.get("ENABLE_PAID_PIPELINE"):
        raise RuntimeError("真实模型处理未启用，请先配置模型密钥和调用预算")
    directory = Path(job["job_dir"]).resolve()
    runtime = Path(config["APP_DATA_DIR"]).resolve()
    if runtime not in directory.parents:
        raise RuntimeError("任务目录不在独立运行目录中")
    request_path = directory / "pipeline_request.json"
    payload = {key: job[key] for key in ("job_id", "course_id", "course_name", "input_path", "job_dir")}
    payload["original_filename"] = job.get("original_filename", "source.pdf")
    payload["resume"] = bool(job.get("resume"))
    payload["config_file"] = config["APP_CONFIG_FILE"]
    request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    update(stage="preparing", progress=3, message="正在启动独立处理进程")
    with (directory / "pipeline.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-X", "utf8", str(ROOT / "scripts/pipeline_worker.py"),
                                    "--request", str(request_path), "--settings-stdin"], cwd=directory,
                                    stdin=subprocess.PIPE, stdout=log, stderr=log, text=True, encoding="utf-8")
        names = {"APP_DATA_DIR", "APP_CONFIG_FILE", "ENABLE_PAID_PIPELINE", "DASHSCOPE_API_KEY", "QWEN_MODEL",
                 "API_MAX_COST_YUAN", "API_MAX_REQUESTS", "DATABASE_MODE", "NEO4J_URI", "NEO4J_USER",
                 "NEO4J_PASSWORD", "NEO4J_DATABASE", "NEO4J_CA_BUNDLE", "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
                 "PINECONE_NAMESPACE", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION", "ENABLE_VECTOR_RETRIEVAL", "PER_JOB_API_BUDGET", "OUTBOUND_HTTP_PROXY"}
        # Effective settings travel through a private process pipe, never argv/logs.
        process.stdin.write(json.dumps({key: config.get(key) for key in names}, ensure_ascii=False))
        process.stdin.close()
        last = None
        try:
            while process.poll() is None:
                if cancelled():
                    process.terminate()
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                    from .jobs import JobCancelled
                    raise JobCancelled("任务已取消，已完成的中间文件与预算记录保留")
                progress_file = directory / "pipeline_progress.json"
                if progress_file.exists():
                    try:
                        value = json.loads(progress_file.read_text(encoding="utf-8"))
                        if value != last:
                            update(**value)
                            last = value
                    except (OSError, ValueError):
                        pass
                time.sleep(0.4)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=8)
    result_path = directory / "pipeline_result.json"
    if process.returncode != 0 or not result_path.is_file():
        error_file = directory / "pipeline_error.json"
        if error_file.is_file():
            info = json.loads(error_file.read_text(encoding="utf-8"))
            raise RuntimeError(info.get("message", "处理失败，请查看任务记录"))
        raise RuntimeError("处理进程未成功结束，中间文件已保留")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    update(stage="publishing", progress=95, message="入库读回完成，正在发布课程图谱")
    return result
