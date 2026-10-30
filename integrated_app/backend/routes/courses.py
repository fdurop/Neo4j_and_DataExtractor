from flask import Blueprint, abort, current_app, g, jsonify, request
from backend.routes.auth import require_teacher

courses_bp = Blueprint("courses", __name__)


@courses_bp.get("/courses")
def get_courses():
    return jsonify(success=True, courses=current_app.extensions["course_store"].list())


@courses_bp.post("/courses")
@require_teacher
def add_course():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        abort(400, description="请求必须为JSON对象。")
    return jsonify(current_app.extensions["course_store"].add(data.get("name"), g.user.id))


@courses_bp.delete("/courses/<course_id>")
@require_teacher
def delete_course(course_id):
    jobs = current_app.extensions["job_service"].list(course_id)
    if any(item["status"] in {"queued", "running", "cancel_requested"} for item in jobs):
        abort(409, description="课程仍有处理中的任务，请先取消或等待完成。")
    current_app.extensions["course_store"].archive(course_id)
    return jsonify(success=True, archived=True)
