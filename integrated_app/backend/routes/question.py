"""Course-aware question endpoint with explicit unavailable/error responses."""
from flask import Blueprint, abort, current_app, jsonify, request
from backend.routes.auth import require_login

qa_bp = Blueprint("qa", __name__)


@qa_bp.post("/")
@require_login
def qa():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("question"), str) or not data["question"].strip():
        abort(400, description="请输入问题。")
    if len(data["question"]) > 12000:
        abort(400, description="问题过长。")
    course = current_app.extensions["course_store"].resolve(data.get("course_id"), data.get("course_name"))
    if data.get("job_id"):
        selected_job = current_app.extensions["job_service"].get(data["job_id"])
        if selected_job["course_id"] != course["id"]:
            abort(400, description="课件任务不属于当前课程。")
        course = dict(course, retrieval_job_id=selected_job["id"])
    runner = current_app.config.get("QA_RUNNER")
    if runner is None:
        try:
            from backend.services.qa import answer_question
        except ImportError:
            abort(503, description="课程问答尚未配置。")
        runner = answer_question
    try:
        result = runner(data["question"].strip(), course)
    except (RuntimeError, ValueError) as error:
        abort(503, description=getattr(error, "public_message", "课程问答未就绪，请检查检索配置和课程数据。"))
    if isinstance(result, str):
        result = {"answer": result}
    if not isinstance(result, dict) or not result.get("answer"):
        abort(502, description="模型未返回有效回答。")
    return jsonify(success=True, **result)
