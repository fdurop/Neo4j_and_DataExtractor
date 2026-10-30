"""Isolated PDF -> budgeted models -> course-scoped Neo4j -> graph export."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engines/extraction"))
sys.path.insert(0, str(ROOT / "engines/extraction/src2"))
sys.path.insert(0, str(ROOT / "engines/graph"))
from app_config import load_settings, vision_config
from validation_budget import ValidationBudget, BudgetedQwenJSONClient
from validation_vision_client import BudgetedVisionClient
from validate_full_document_api import prepare_document, execute_prepared
from validate_fresh_course_unit import run_graph_phase, graph_config
import all_builder


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute(job, settings):
    directory = Path(job["job_dir"]).resolve()
    pdf = Path(job["input_path"]).resolve()
    runtime = Path(settings["APP_DATA_DIR"]).resolve()
    if runtime not in directory.parents or directory not in pdf.parents:
        raise ValueError("任务文件归属不正确")
    original_name = str(job.get("original_filename") or "source.pdf").replace("\\", "/").rsplit("/", 1)[-1]
    original_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", original_name)
    original_name = (original_name[:150].rstrip(" .") or "source")
    if not original_name.lower().endswith(".pdf"):
        original_name += ".pdf"
    documents = directory / "documents"
    documents.mkdir(exist_ok=True)
    displayed_pdf = documents / original_name
    if displayed_pdf.exists():
        raise ValueError("当前任务已有处理输入，请创建新任务以保留原结果")
    shutil.copyfile(pdf, displayed_pdf)
    pdf = displayed_pdf
    if not settings["ENABLE_PAID_PIPELINE"] or not settings["DASHSCOPE_API_KEY"]:
        raise ValueError("模型配置缺失或未启用真实处理")
    if settings["DATABASE_MODE"] != "neo4j":
        raise ValueError("完整真实处理需要配置Neo4j数据库，local模式仅供接口测试")
    os.environ["APP_NEO4J_DATABASE"] = settings.get("NEO4J_DATABASE") or ""
    os.environ["APP_NEO4J_CA_BUNDLE"] = settings.get("NEO4J_CA_BUNDLE") or ""

    def progress(stage, percent, message):
        write(directory / "pipeline_progress.json", {"stage": stage, "progress": percent, "message": message})

    progress("preparing", 5, "正在解析课件并准备逐页任务")
    extraction = directory / "extraction"
    manifest = prepare_document(pdf, extraction, course_name=job["course_name"], course_id=job["course_id"], job_id=job["job_id"])
    if not isinstance(manifest, dict):
        manifest = json.loads((extraction / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("preparation_errors"):
        raise RuntimeError("课件解析不完整，已停止模型调用")
    ledger = ValidationBudget(runtime / "api_budget.json", max_cost_yuan=settings["API_MAX_COST_YUAN"],
                              max_requests=settings["API_MAX_REQUESTS"], model=settings["QWEN_MODEL"])
    plan_count = len(manifest["tasks"])
    if plan_count > 65:
        raise RuntimeError("当前课件视觉任务超过单阶段65次上限，尚未调用模型，请拆分课件")
    if plan_count + manifest["original_page_count"] > ledger.snapshot()["remaining_requests"]:
        raise RuntimeError("剩余请求预算不足以完成当前课件，请减少测试范围或检查预算")
    # Check credentials/TLS/database availability before incurring model costs.
    from neo4j_runtime import driver_for
    driver = driver_for(settings["NEO4J_URI"], settings["NEO4J_USER"], settings["NEO4J_PASSWORD"],
                        settings.get("NEO4J_DATABASE"), settings.get("NEO4J_CA_BUNDLE"))
    try:
        driver.verify_connectivity()
    finally:
        driver.close()
    config = vision_config(settings)

    class Vision(BudgetedVisionClient):
        completed = 0

        def _chat_with_image(self, *args, **kwargs):
            text = super()._chat_with_image(*args, **kwargs)
            self.completed += 1
            progress("extracting", min(55, 10 + int(45 * self.completed / max(1, plan_count))),
                     f"已处理 {self.completed} 个视觉/结构任务")
            return text

    vision = Vision(config, ledger, extraction, stage="vision_" + job["job_id"], max_stage_requests=65)
    execution = execute_prepared(extraction, vision, include_specialists=True)
    if execution.get("status") != "success":
        raise RuntimeError("提取阶段未完整完成，已停止建图，详情见任务产物")
    progress("building_graph", 60, "素材提取完成，正在抽取概念与关系")
    graph_output = directory / "graph"

    class GraphClient(BudgetedQwenJSONClient):
        completed = 0

        def chat_json(self, *args, **kwargs):
            value = super().chat_json(*args, **kwargs)
            self.completed += 1
            progress("building_graph", 65, f"建图阶段已完成 {self.completed} 次模型请求")
            return value

    client = GraphClient(config, ledger, graph_output, "graph_" + job["job_id"], max_stage_requests=35, max_output_tokens=8192)
    provenance = {"source_type": "fresh_original_pdf_extraction", "source_pdf": str(pdf),
                  "source_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                  "extraction_manifest": str(extraction / "execution_manifest.json"), "extraction_run_id": job["job_id"]}
    summary = run_graph_phase(extraction / "kg_data", execution["document_id"], graph_output, client, provenance,
                              expected_pages=manifest["original_page_count"], max_stage_requests=35)
    if summary.get("status") != "success":
        raise RuntimeError("建图阶段未完整完成，已停止入库，成功任务与失败记录均保留")
    progress("database_import", 80, "正在按课程写入Neo4j")
    config_db = graph_config(graph_output)
    config_db.neo4j_uri, config_db.neo4j_user, config_db.neo4j_password = (
        settings["NEO4J_URI"], settings["NEO4J_USER"], settings["NEO4J_PASSWORD"])
    config_db.data_version = job["job_id"]
    formal = all_builder._load_formal_data(config_db)
    if formal["manifest"]["course_id"] != job["course_id"]:
        raise RuntimeError("建图课程身份与上传任务不一致，停止入库")
    knowledge = json.loads(Path(config_db.final_concepts_path).read_text(encoding="utf-8"))
    database_report = all_builder._upload_to_neo4j(formal, knowledge, config_db)
    progress("database_readback", 90, "正在读回课程图谱并核对归属")
    graph = all_builder.export_graph_from_neo4j(str(directory / "database_graph.json"), radius=0,
                course_id=job["course_id"], neo4j_uri=settings["NEO4J_URI"], neo4j_user=settings["NEO4J_USER"],
                neo4j_password=settings["NEO4J_PASSWORD"])
    if not graph.get("nodes") or any(node.get("properties", {}).get("course_id") != job["course_id"] for node in graph["nodes"]):
        raise RuntimeError("数据库读回为空或含其他课程节点，未发布图谱")
    graph["integration"] = {"course_id": job["course_id"], "course_name": job["course_name"], "job_id": job["job_id"],
                             "source_url": "/api/jobs/" + job["job_id"] + "/source", "fresh_extraction": True,
                             "teacher_review_required": True}
    result = {"graph": graph, "artifacts": {"extraction": "extraction", "graph": "graph", "database_export": "database_graph.json"},
              "database": database_report, "budget": ledger.snapshot(), "completed_at": datetime.now(timezone.utc).isoformat()}
    write(directory / "pipeline_result.json", result)
    progress("publishing", 95, "已读回课程图谱，等待发布审核快照")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--settings-stdin", action="store_true")
    args = parser.parse_args()
    job = json.loads(args.request.read_text(encoding="utf-8"))
    settings = json.load(sys.stdin) if args.settings_stdin else load_settings(job["config_file"])
    try:
        execute(job, settings)
    except Exception as error:
        # No raw driver/provider exception or key is emitted to the job API.
        safe = str(error) if isinstance(error, (RuntimeError, ValueError)) else "处理失败，请查看阶段记录（" + type(error).__name__ + "）"
        for key in ("DASHSCOPE_API_KEY", "NEO4J_PASSWORD"):
            if settings.get(key):
                safe = safe.replace(settings[key], "[REDACTED]")
        write(Path(job["job_dir"]) / "pipeline_error.json", {"error_type": type(error).__name__, "message": safe[:600]})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
