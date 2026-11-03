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
    from backend.services.course_delete import delete_course as purge
    from backend.routes.graph_contract import GraphError
    try:
        result = purge(current_app, course_id)
    except GraphError as error:
        abort(error.status, description=str(error))
    return jsonify(success=True, deleted=True, **result)
