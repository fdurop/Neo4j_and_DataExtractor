"""Explicit course deletion. No network or data mutation on import/startup.

Provider operations are idempotent. Keep the catalogue's deleting marker until
every store is cleaned, so a failed request/restart can be retried safely.
Historical backups and shared API cost ledgers are not active course stores.
"""
import hashlib
import json
from pathlib import Path
import shutil

from werkzeug.exceptions import Conflict, NotFound, ServiceUnavailable

from .course_store import atomic_json, read_json


def _safe_folder(path, root):
    root = Path(root).resolve()
    path = Path(path)
    # Never follow a junction/symlink out of the application's data directory.
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise Conflict("课程资料目录包含链接，未执行删除，请检查运行目录。")
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise Conflict("课程资料目录超出运行目录，未执行删除。")
    return resolved


def _remove_folder(path, root):
    target = _safe_folder(path, root)
    if target.exists():
        # Validate descendants too: do not traverse externally linked assets.
        for child in target.rglob("*"):
            _safe_folder(child, root)
        shutil.rmtree(target)


def _native_delete(repo, identity, *, check_only=False):
    # Course roots in older data may have id but no course_id. Revision history
    # uses owner_course_id, and must also go or deleted content remains undoable.
    selected = "coalesce((n.course_id=$id OR (n:Course AND n.id=$id) OR (n:GraphRevision AND n.owner_course_id=$id)),false)"
    peer_selected = selected.replace("n.", "peer.").replace("n:", "peer:")
    driver = repo._driver()
    try:
        with driver.session() as session:
            with session.begin_transaction() as tx:
                if not check_only:
                    # Serialize with the repository's revision-checked writer.
                    tx.run("MATCH (c:Course {id:$id}) SET c.review_revision=coalesce(c.review_revision,0)", id=identity).consume()
                invalid = tx.run("MATCH (n) WHERE " + selected +
                    " AND ((n.course_id IS NOT NULL AND n.course_id<>$id) OR (n:Course AND coalesce(n.id,'')<>$id) "
                    "OR (n:GraphRevision AND coalesce(n.owner_course_id,'')<>$id)) "
                    "RETURN count(n) AS count", id=identity).single()
                outside = tx.run("MATCH (n)-[r]-(peer) WHERE " + selected + " AND (NOT " + peer_selected +
                    " OR (r.course_id IS NOT NULL AND r.course_id<>$id)) RETURN count(r) AS count", id=identity).single()
                if invalid["count"] or outside["count"]:
                    raise Conflict("课程存在归属异常的节点或跨课程关系，未删除，请先核查数据。")
                if not check_only:
                    count = tx.run("MATCH (n) WHERE " + selected +
                        " WITH n DETACH DELETE n RETURN count(*) AS count", id=identity).single()["count"]
                    tx.commit()
                    return count
    finally:
        driver.close()


def _delete_vectors(settings, identity, manifest):
    from . import vector_service
    configured = bool(settings.get("PINECONE_API_KEY") and settings.get("PINECONE_INDEX_NAME"))
    if not (configured or manifest or settings.get("ENABLE_VECTOR_RETRIEVAL")):
        return
    # Clean both the current destination and the last recorded destination if
    # configuration changed. Always use metadata filtering, never deleteAll.
    destinations = {(settings.get("PINECONE_INDEX_NAME"), settings.get("PINECONE_NAMESPACE", ""),
                     int(settings.get("EMBEDDING_DIMENSION", 1024)))}
    if manifest:
        destinations.add((manifest.get("index"), manifest.get("namespace", ""), int(manifest.get("dimension", 1024))))
    with vector_service._lock:
        for index, namespace, dimension in sorted(destinations, key=str):
            connection = dict(settings, PINECONE_INDEX_NAME=index, EMBEDDING_DIMENSION=dimension)
            host, headers = vector_service._connection(connection)
            vector_service._post(host + "/vectors/delete", headers,
                {"namespace": namespace, "filter": {"course_id": {"$eq": identity}}})


def delete_course(app, identity):
    store = app.extensions["course_store"]
    # The HTTP lifecycle lock also covers uploads, retries, editing and Q&A.
    with store.operation_lock(identity):
        course = next((c for c in store.list() if c["id"] == identity), None)
        if course is None:
            raise NotFound("课程不存在或已删除。")
        jobs = app.extensions["job_service"]
        records = jobs.list(identity, include_hidden=True)
        with jobs.lock:
            if any(item["status"] in {"queued", "running", "cancel_requested"} or
                   (jobs.futures.get(item["id"]) is not None and not jobs.futures[item["id"]].done())
                   for item in records):
                raise Conflict("课程仍有处理中的任务，请先取消或等待完成。")
        root = Path(app.config["APP_DATA_DIR"]).resolve()
        repo = app.extensions.get("graph_repository")
        # Factory-only/test applications may use the original JSON graph store.
        if repo is None:
            from .graph_repository import GraphRepository
            repo = GraphRepository(app.config)
        with repo.lock:
            return _delete_locked(app, store, jobs, records, root, repo, course)


def _delete_locked(app, store, jobs, records, root, repo, course):
    identity = course["id"]
    job_ids = sorted(set(course.get("deletion_job_ids", [])) | {item["id"] for item in records})
    folders = [jobs._path(job_id).parent for job_id in job_ids]
    folders.append(root / "vector_indexes" / hashlib.sha256(identity.encode("utf-8")).hexdigest())
    # Verify exact absolute targets before any recursive deletion/provider call.
    for folder in folders:
        _safe_folder(folder, root)
        if folder.exists():
            for child in folder.rglob("*"):
                _safe_folder(child, root)
    manifest = read_json(folders[-1] / "manifest.json", {})
    cached = read_json(repo.records_path, {})
    graphs = read_json(repo.path, {})
    if not isinstance(cached, dict) or not isinstance(graphs, dict) or not isinstance(manifest, dict):
        raise Conflict("课程存储格式异常，未执行删除。")
    neo4j = app.config.get("DATABASE_MODE") == "neo4j"
    try:
        if neo4j:
            _native_delete(repo, identity, check_only=True)
    except Conflict:
        raise
    except Exception:
        app.logger.exception("Course deletion preflight failed for %s", identity)
        raise ServiceUnavailable("数据库检查未完成，课程尚未删除，请稍后重试。") from None

    store.mark_deleting(identity, job_ids)
    try:
        _delete_vectors(dict(app.config), identity, manifest)
        node_count = _native_delete(repo, identity) if neo4j else len(cached.get(identity, {}).get("graph", {}).get("nodes", []))
        # Reload shared caches: other courses may have been read while preflighting.
        cached = read_json(repo.records_path, {})
        cached.pop(identity, None)
        atomic_json(repo.records_path, cached)
        graphs = read_json(repo.path, {})
        graph = graphs.get(course["name"], {})
        graph_owner = graph.get("integration", {}).get("course_id")
        if graph_owner in (None, identity):
            graphs.pop(course["name"], None)
        atomic_json(repo.path, graphs)
        repo._map_metadata_cache.pop(identity, None)
        for folder in folders:
            _remove_folder(folder, root)
        with jobs.lock:
            for job_id in job_ids:
                jobs.futures.pop(job_id, None)
        # Remove catalogue last; on any earlier failure the UI can retry.
        store.remove(identity)
        return {"deleted_nodes": node_count, "deleted_uploads": len(job_ids)}
    except Exception:
        app.logger.exception("Course deletion incomplete for %s; retry remains available", identity)
        raise ServiceUnavailable("删除尚未完成，课程已暂停使用。请重试删除；已清理的部分不会重复创建。") from None
