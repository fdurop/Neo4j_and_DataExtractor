"""Course-scoped graph editing backed by local JSON, without model/DB calls."""
import json
import gzip
import time
import os
import tempfile
from threading import RLock

from flask import Blueprint, current_app, jsonify, request, session

from . import graph_contract
from .graph_contract import GraphError


kg_bp = Blueprint("kg", __name__)
BASE_DIR = os.path.dirname(__file__)
GRAPH_PATH = os.path.join(BASE_DIR, "graphs.json")
_write_lock = RLock()
from backend.services.graph_load_progress import LoadProgress
_load_progress = LoadProgress()


def _graph_path():
    return current_app.config.get("KG_GRAPH_PATH", GRAPH_PATH)


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise GraphError("图谱文件读取失败，请检查文件格式和权限", 409) from exc


def save_json(path, data):
    """Replace a complete JSON file atomically; keep old data on write failure."""
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary = tempfile.mkstemp(prefix=".graphs-", suffix=".json.tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def load_graphs():
    if "graph_repository" in current_app.extensions:
        return current_app.extensions["graph_repository"].load_all()
    graphs = load_json(_graph_path(), {})
    if not isinstance(graphs, dict):
        raise GraphError("图谱文件需要按课程名称保存 JSON 对象", 409)
    return graphs


def save_graphs(data):
    save_json(_graph_path(), data)


@kg_bp.errorhandler(GraphError)
def graph_error(error):
    return jsonify({"success": False, "error": str(error)}), error.status


def _mutate(operation, *arguments):
    if "graph_repository" in current_app.extensions:
        repo = current_app.extensions["graph_repository"]
        identity = request.args.get("course_id")
        if arguments and isinstance(arguments[0], dict):
            payload = dict(arguments[0])
            course = repo.resolve(course_id=identity, name=payload.get("course_name") or request.args.get("course_name"))
            payload["course_name"] = course["name"]
            arguments = (payload, *arguments[1:])
        elif operation is graph_contract.delete_node:
            course = repo.resolve(course_id=identity, name=arguments[1])
            arguments = (arguments[0], course["name"])
        return jsonify(repo.mutate(operation, *arguments, expected_revision=_expected_revision()))
    # This lock protects concurrent requests in this process, not across workers.
    with _write_lock:
        output, result = operation(load_graphs(), *arguments)
        graph = output.get(result.get("course_name"), {})
        if "node_count" in graph:
            graph["node_count"] = len(graph.get("nodes", []))
        if "link_count" in graph:
            graph["link_count"] = len(graph.get("links", []))
        save_graphs(output)
    return jsonify(result)


@kg_bp.route("/graph", methods=["GET"])
def get_graph():
    """Return all stored metadata, including sources and teacher review states."""
    course = request.args.get("course_name")
    if "graph_repository" in current_app.extensions:
        repo = current_app.extensions["graph_repository"]
        if request.args.get("view") == "map":
            started = time.perf_counter()
            token=request.args.get("progress_id")
            progress=_load_progress.start(token,session.get("user_id")) if token else None
            try:
                result = repo.get_map(name=course, course_id=request.args.get("course_id"), **({"progress":progress} if progress else {}))
                if progress:progress(phase="transfer",nodes=len(result["nodes"]),links=len(result["links"]))
                response = _graph_response(result)
                if progress:progress(phase="done")
            except Exception:
                if progress:progress(phase="error")
                raise
            response.headers["Server-Timing"] = "graph;dur=" + str(round((time.perf_counter()-started)*1000, 1))
            return response
        return jsonify(repo.get(name=course, course_id=request.args.get("course_id")))
    return jsonify(load_graphs().get(course, {"nodes": [], "links": []}))


@kg_bp.get("/graph/load-progress/<token>")
def get_load_progress(token):
    response=jsonify(_load_progress.get(token,session.get("user_id")))
    response.headers["Cache-Control"]="no-store"
    return response


def _graph_response(result):
    response = jsonify(result)
    response.headers["Cache-Control"] = "no-store"
    response.vary.add("Accept-Encoding")
    if request.accept_encodings["gzip"] > 0:
        response.set_data(gzip.compress(response.get_data(), compresslevel=4))
        response.headers["Content-Encoding"] = "gzip"
    return response


@kg_bp.get("/graph/node/<path:node_id>/details")
def get_node_details(node_id):
    repo = current_app.extensions.get("graph_repository")
    if repo is None: raise GraphError("当前图谱不支持按需加载", 409)
    revision = request.args.get("revision", type=int)
    if request.args.get("revision") is not None and (revision is None or revision < 0):
        raise GraphError("图谱版本无效", 400)
    return _graph_response(repo.get_details(node_id, course_id=request.args.get("course_id"),
        name=request.args.get("course_name"), expected_revision=revision))


@kg_bp.before_request
def check_graph_access():
    if current_app.config.get("REQUIRE_AUTH", True) and not session.get("user_id"):
        raise GraphError("请先登录", 401)
    if request.method != "GET" and current_app.config.get("REQUIRE_AUTH", True) and session.get("role") != "teacher":
        raise GraphError("此操作需要教师身份", 403)
    payload = request.get_json(silent=True)
    if payload is not None and not isinstance(payload, dict):
        raise GraphError("请求必须是 JSON 对象", 400)
    payload = payload or {}
    query_name = request.args.get("course_name")
    if query_name and payload.get("course_name") and query_name != payload["course_name"]:
        raise GraphError("课程标识不匹配", 404)
    if "graph_repository" in current_app.extensions and request.args.get("course_id"):
        current_app.extensions["graph_repository"].resolve(
            course_id=request.args["course_id"], name=payload.get("course_name") or request.args.get("course_name"))


@kg_bp.route("/graph/node", methods=["POST"])
def add_node():
    return _mutate(graph_contract.add_node, request.get_json(silent=True))


@kg_bp.route("/graph/node", methods=["PUT"])
def edit_node():
    return _mutate(graph_contract.edit_node, request.get_json(silent=True))


@kg_bp.route("/graph/node/<path:node_id>", methods=["DELETE"])
def delete_node(node_id):
    return _mutate(graph_contract.delete_node, node_id, request.args.get("course_name"))


@kg_bp.route("/graph/link", methods=["POST"])
def add_link():
    return _mutate(graph_contract.add_link, request.get_json(silent=True))


@kg_bp.route("/graph/link", methods=["PUT"])
def edit_link():
    return _mutate(graph_contract.edit_link, request.get_json(silent=True))


@kg_bp.route("/graph/link", methods=["DELETE"])
def delete_link():
    return _mutate(graph_contract.delete_link, request.get_json(silent=True))


def _expected_revision():
    value = request.headers.get("X-Graph-Revision")
    if value is None:
        return None
    try:
        revision = int(value)
        if revision < 0:
            raise ValueError()
        return revision
    except ValueError:
        raise GraphError("图谱版本无效。", 400)


@kg_bp.post("/graph/undo")
def undo_graph():
    repo = current_app.extensions.get("graph_repository")
    if repo is None:
        raise GraphError("当前存储不支持撤销。", 409)
    payload = request.get_json(silent=True) or {}
    course = repo.resolve(course_id=request.args.get("course_id"), name=payload.get("course_name") or request.args.get("course_name"))
    expected = _expected_revision()
    if expected is None:
        raise GraphError("撤销需要当前版本号。", 400)
    return jsonify(repo.undo(course, expected))
