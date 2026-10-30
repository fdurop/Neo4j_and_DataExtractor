"""Review published material descriptions while preserving graph evidence."""
from pathlib import Path
from urllib.parse import quote

from flask import Blueprint, abort, current_app, jsonify, request, send_file
from PIL import Image
from werkzeug.exceptions import NotFound

from backend.routes.auth import require_login, require_teacher
from backend.routes import graph_contract
from backend.routes.graph_contract import GraphError

repair_bp = Blueprint("repair", __name__)
ROOT = Path(__file__).resolve().parents[2]
MATERIAL_TYPES = {"figure", "table", "formula"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@repair_bp.errorhandler(GraphError)
def graph_error(error):
    return jsonify(success=False, error=str(error)), error.status


def repository():
    result = current_app.extensions.get("graph_repository")
    if result is None:
        abort(503, description="课程图谱存储尚未配置。")
    return result


def kind(node):
    value = graph_contract.value_of(node, "type") or graph_contract.value_of(node, "node_type", "")
    return str(value).lower()


def properties(node):
    value = node.get("properties")
    return value if isinstance(value, dict) else {}


def job_record(job_id):
    if not isinstance(job_id, str):
        return None
    try:
        return current_app.extensions["job_service"].get(job_id, private=True)
    except NotFound:
        return None


def image_is_valid(path):
    if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG"}:
                return False
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return False


def image_for(node, page, job):
    if job is None:
        return None
    job_dir = Path(job["job_dir"]).resolve()
    assets = (job_dir / "extraction/assets").resolve()
    if not assets.is_relative_to(job_dir):
        return None
    candidates = []
    for record in (properties(node), properties(page or {})):
        for field in ("enhanced_path", "source_path", "page_image_path", "image_path"):
            value = record.get(field)
            if not isinstance(value, str) or not value:
                continue
            path = Path(value)
            if path.is_absolute():
                candidates.append(path)
            else:
                # The copied exporter serializes paths relative to its
                # WORKSPACE_DIR (integrated_app/engines), so real results can
                # begin '../runtime/...'. Every candidate must still resolve
                # inside this exact job's assets; no basename remapping.
                candidates.extend((ROOT / "engines" / path, ROOT / path,
                                   job_dir / path, job_dir / "extraction" / path, assets / path))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except (OSError, ValueError):
            continue
        if resolved.is_relative_to(assets) and image_is_valid(resolved):
            relative = resolved.relative_to(assets).as_posix()
            return "/api/jobs/" + job["id"] + "/assets/" + quote(relative, safe="/")
    return None


def course_graphs(values):
    repo = repository()
    identity, name = values.get("course_id"), values.get("course_name")
    if identity or name:
        course = repo.resolve(course_id=identity, name=name)
        return [(course, repo.get(course_id=course["id"]))]
    graphs = repo.load_all()
    return [(course, graphs[course["name"]]) for course in current_app.extensions["course_store"].list()
            if course["name"] in graphs]


def material_items(course, graph, *, pending_only=True):
    page_nodes = {}
    for node in graph.get("nodes", []):
        if kind(node) == "page":
            for identity in (node.get("id"), properties(node).get("page_id")):
                if identity:
                    page_nodes[identity] = node
    result = []
    for node in graph.get("nodes", []):
        if kind(node) not in MATERIAL_TYPES:
            continue
        review = graph_contract.value_of(node, "review_status", "pending")
        if pending_only and review in {"approved", "rejected"}:
            continue
        data = properties(node)
        page = page_nodes.get(data.get("source_page_id") or data.get("page_id"))
        page_data = properties(page or {})
        job = job_record(data.get("job_id") or page_data.get("job_id"))
        if job and job.get("course_id") != course["id"]:
            job = None
        # Do not use the latest graph-level job for an older material node.
        # A Page with a matching ID supplies the original page position.
        page_no = page_data.get("page_no")
        if not isinstance(page_no, int) or isinstance(page_no, bool) or page_no < 1:
            page_no = None
        source_url = job["source_url"] if job else None
        if source_url and page_no:
            source_url += "#page=" + str(page_no)
        description = graph_contract.value_of(node, "description", "")
        item = {"id": node.get("id"), "course_id": course["id"], "course_name": course["name"],
                "name": graph_contract.value_of(node, "name", node.get("id")), "type": kind(node),
                "description": description or data.get("latex") or "", "review_status": review,
                "page_no": page_no, "source_url": source_url,
                "description_scope": data.get("description_scope"),
                "semantic_review_flags": data.get("semantic_review_flags", [])}
        image_url = image_for(node, page, job)
        if image_url:
            item["image_url"] = image_url
        result.append(item)
    return result


@repair_bp.get("/repair/items")
@require_teacher
def get_repair_items():
    items = [item for course, graph in course_graphs(request.args) for item in material_items(course, graph)]
    return jsonify(success=True, items=items)


@repair_bp.put("/repair/items/<path:item_id>")
@require_teacher
def update_repair_item(item_id):
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or set(data) - {"course_id", "course_name", "description"}:
        abort(400, description="仅支持课程身份和素材文案字段。")
    description = data.get("description")
    if not isinstance(description, str) or len(description) > 20000:
        abort(400, description="素材文案需为不超过20000字符的字符串。")
    matches = [(course, item) for course, graph in course_graphs(data)
               for item in material_items(course, graph, pending_only=False) if item["id"] == item_id]
    if not matches:
        abort(404, description="课程中未找到该素材。")
    if len(matches) != 1:
        abort(409, description="素材编号存在歧义，请指定课程编号。")
    course, _ = matches[0]
    result = repository().mutate(graph_contract.edit_node, {"course_name": course["name"], "id": item_id,
                                  "description": description, "origin": "teacher_edit"})
    result["message"] = "素材文案已更新，相关图谱关系仍需核对"
    return jsonify(result)


@repair_bp.get("/jobs/<job_id>/assets/<path:asset>")
@require_login
def get_job_asset(job_id, asset):
    job = job_record(job_id)
    if job is None or not asset or "\\" in asset or "\x00" in asset:
        abort(404)
    job_dir = Path(job["job_dir"]).resolve()
    try:
        assets = (job_dir / "extraction/assets").resolve()
        candidate = (assets / asset).resolve()
    except (OSError, ValueError):
        abort(404)
    if not assets.is_relative_to(job_dir):
        abort(404)
    if not candidate.is_relative_to(assets) or not image_is_valid(candidate):
        abort(404)
    response = send_file(candidate, mimetype="image/png" if candidate.suffix.lower() == ".png" else "image/jpeg")
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response
