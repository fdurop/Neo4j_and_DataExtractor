"""Authoritative reviewed graph snapshots with atomic Neo4j revisions."""
import copy
import logging
import time
from neo4j.exceptions import ServiceUnavailable, SessionExpired
import base64
import zlib
import json
from pathlib import Path
from threading import RLock
import sys
from urllib.parse import urlencode
from datetime import datetime, timezone

from flask import current_app
from backend.routes.graph_contract import GraphError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines/graph"))
from neo4j_runtime import driver_for, clean_snapshot_graph
from .graph_merge import merge_review_graph
from .native_graph import apply_delta, read_graph, encode_snapshot, decode_snapshot, validate_course_graph


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class _DriverLease:
    """Sessions are short lived; the thread-safe driver/pool belongs to the repository."""
    def __init__(self, driver): self.driver = driver
    def __getattr__(self, name): return getattr(self.driver, name)
    def close(self): pass


class GraphRepository:
    def __init__(self, config):
        self.config = dict(config)
        self.path = Path(config["KG_GRAPH_PATH"])
        self.records_path = self.path.with_name("graph_records.json")
        self.lock = RLock()
        self._pool_lock = RLock()
        self.cache_warning = None
        self._driver_instance = None
        self._map_metadata_cache = {}

    def courses(self):
        path = Path(self.config["APP_DATA_DIR"]) / "courses.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def resolve(self, name=None, course_id=None):
        candidates = [c for c in self.courses() if not c.get("archived") and not c.get("deleting")
                      and (not course_id or c.get("id") == course_id or c.get("course_id") == course_id)
                      and (not name or c.get("name") == name)]
        if (not name and not course_id) or len(candidates) != 1:
            raise GraphError("课程不存在或课程标识不匹配", 404)
        return candidates[0]

    def _driver(self):
        with self._pool_lock:
            if self._driver_instance is None:
                c = self.config
                self._driver_instance = driver_for(c["NEO4J_URI"], c["NEO4J_USER"], c["NEO4J_PASSWORD"],
                    c.get("NEO4J_DATABASE"), c.get("NEO4J_CA_BUNDLE"))
            return _DriverLease(self._driver_instance)

    def close(self):
        with self._pool_lock:
            if self._driver_instance is not None:
                self._driver_instance.close()
                self._driver_instance = None

    def _map_metadata(self, session, course_id, revision):
        cached = self._map_metadata_cache.get(course_id)
        if cached and cached["revision"] == revision:
            return copy.deepcopy(cached["metadata"])
        graph = None
        # Only graph-level audit metadata may use a revision-checked local copy.
        # Native node/relationship existence and editable values are always queried live.
        try:
            record = json.loads(self.records_path.read_text(encoding="utf-8")).get(course_id, {})
            if record.get("revision") == revision:
                graph = record.get("graph")
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        if graph is None:
            row = session.run("MATCH (c:Course {id:$id}) RETURN c.review_graph_json AS review, "
                "coalesce(c.review_revision,0) AS revision", id=course_id).single()
            if row is None or row["revision"] != revision:
                raise GraphError("图谱已更新，请重新加载", 409)
            graph = decode_snapshot(row["review"])
        metadata = {k:v for k,v in graph.items() if k not in {"nodes", "links"}}
        self._map_metadata_cache[course_id] = {"revision": revision, "metadata": copy.deepcopy(metadata)}
        return metadata

    def get_map(self, *, course_id=None, name=None, _attempt=0, progress=None):
        from .graph_projection import project_stored, read_map
        course = self.resolve(name=name, course_id=course_id)
        if self.config.get("DATABASE_MODE") != "neo4j":
            if progress:progress(phase="snapshot")
            result = project_stored(self.get(course_id=course["id"]))
            result["details_lazy"] = True
            return result
        driver = self._driver()
        try:
            with driver.session(default_access_mode="READ", fetch_size=10000) as session:
                row = session.run("MATCH (c:Course {id:$id}) RETURN CASE WHEN c.native_graph_version IS NULL THEN c.review_graph_json ELSE null END AS review, "
                    "coalesce(c.review_revision,0) AS revision,c.native_graph_version AS native_version", id=course["id"]).single()
                if row is None:
                    return {"nodes": [], "links": [], "projection": "map-v1", "review_revision": 0}
                if row["native_version"]:
                    result = self._map_metadata(session, course["id"], row["revision"])
                    result["nodes"], result["links"] = read_map(session, course["id"], progress=progress) if progress else read_map(session, course["id"])
                    result["projection"] = "map-v1"
                    valid = {n["id"] for n in result["nodes"]}
                    result["hierarchy_candidates"] = [x for x in result.get("hierarchy_candidates", [])
                        if x.get("source") in valid and x.get("target") in valid]
                else:
                    if progress:progress(phase="snapshot")
                    result = project_stored(decode_snapshot(row["review"]))
                result.update(review_revision=row["revision"], details_lazy=True,
                              node_count=len(result["nodes"]), link_count=len(result["links"]))
                latest = session.run("MATCH (c:Course {id:$id}) RETURN coalesce(c.review_revision,0) AS revision", id=course["id"]).single()
                if latest is None or latest["revision"] != row["revision"]:
                    raise GraphError("图谱在读取时已更新，请重新加载", 409)
                return result
        except GraphError as error:
            if error.status == 409 and _attempt == 0:
                return self.get_map(course_id=course["id"], _attempt=1, progress=progress)
            raise
        except Exception as error:
            logging.getLogger(__name__).warning("Map read failed: %s", type(error).__name__)
            raise GraphError("图谱加载失败，请检查数据库连接后重新加载", 503) from None
        finally:
            driver.close()

    def get_details(self, node_id, *, course_id=None, name=None, expected_revision=None):
        from .graph_projection import stored_details, read_details
        course = self.resolve(name=name, course_id=course_id)
        with self.lock:
            if self.config.get("DATABASE_MODE") != "neo4j":
                graph = self.get(course_id=course["id"])
                revision = graph.get("review_revision", 0)
                result = stored_details(graph, node_id)
            else:
                driver = self._driver()
                try:
                    with driver.session(default_access_mode="READ") as session:
                        row = session.run("MATCH (c:Course {id:$id}) RETURN coalesce(c.review_revision,0) AS revision, "
                            "c.native_graph_version AS native_version, CASE WHEN c.native_graph_version IS NULL "
                            "THEN c.review_graph_json ELSE null END AS review", id=course["id"]).single()
                        if row is None: raise GraphError("课程图谱不存在", 404)
                        revision = row["revision"]
                        if expected_revision is not None and revision != expected_revision:
                            raise GraphError("图谱已更新，请重新加载后查看资料", 409)
                        result = read_details(session, course["id"], node_id) if row["native_version"] else stored_details(decode_snapshot(row["review"]), node_id)
                except GraphError:
                    raise
                except Exception as error:
                    logging.getLogger(__name__).warning("Details read failed: %s", type(error).__name__)
                    raise GraphError("资料加载失败，请重试", 503) from None
                finally:
                    driver.close()
            if expected_revision is not None and revision != expected_revision:
                raise GraphError("图谱已更新，请重新加载后查看资料", 409)
            result["review_revision"] = revision
            return result

    def _records(self, course_id=None, review_only=False, _attempt=0):
        if self.config.get("DATABASE_MODE") != "neo4j":
            records = json.loads(self.records_path.read_text(encoding="utf-8")) if self.records_path.exists() else {}
            return {key: value for key, value in records.items() if not course_id or key == course_id}
        courses = self.courses()
        ids = [c["id"] for c in courses if not c.get("archived") and (not course_id or c["id"] == course_id)]
        if not ids:
            return {}
        driver = self._driver()
        try:
            with driver.session(default_access_mode="READ") as session:
                fields = "c.id AS id, c.review_graph_json AS review, coalesce(c.review_revision,0) AS revision, c.native_graph_version AS native_version"
                if not review_only:
                    fields += ", c.model_graph_json AS model, c.previous_review_graph_json AS previous, c.undo_revision AS undo_revision"
                rows = session.run("MATCH (c:Course) WHERE c.id IN $ids RETURN " + fields, ids=ids)
                records = {}
                for r in list(rows):
                    graph = clean_snapshot_graph(decode_snapshot(r["review"])) if r["review"] else {"nodes": [], "links": []}
                    stored_graph = copy.deepcopy(graph)
                    if r["native_version"]:
                        graph = read_graph(session, graph, r["id"])
                        graph["review_revision"] = r["revision"]
                    previous_graph = clean_snapshot_graph(decode_snapshot(r["previous"])) if r.get("previous") else None
                    if not review_only and r.get("undo_revision") is not None:
                        history = session.run("MATCH (h:GraphRevision {owner_course_id:$id, revision:$revision}) RETURN h.before_json_zlib AS before",
                            id=r["id"], revision=r["undo_revision"]).single() if r["undo_revision"] else None
                        if r["undo_revision"] and history is None:
                            raise GraphError("撤销历史缺失，未执行恢复", 409)
                        previous_graph = clean_snapshot_graph(json.loads(zlib.decompress(base64.b64decode(history["before"])))) if history else None
                    records[r["id"]] = {"graph": graph, "stored_graph": stored_graph,
                        "model": clean_snapshot_graph(decode_snapshot(r.get("model") or r["review"])),
                        "revision": r["revision"], "native_version": r["native_version"],
                        "previous_graph": previous_graph}
                return records
        except GraphError:
            raise
        except (ServiceUnavailable, SessionExpired) as error:
            if _attempt < 2:
                driver.close()
                time.sleep(_attempt + 1)
                return self._records(course_id, review_only, _attempt + 1)
            logging.getLogger(__name__).warning("Native read failed: %s", type(error).__name__)
            raise GraphError("数据库图谱读取失败，已保留本地缓存，请检查数据库连接", 503) from None
        except Exception as error:
            logging.getLogger(__name__).warning("Native read failed: %s: %s", type(error).__name__, str(error)[:220])
            raise GraphError("数据库图谱读取失败，已保留本地缓存，请检查数据库连接", 503) from None
        finally:
            driver.close()

    def load_all(self):
        with self.lock:
            records = self._records()
            return {c["name"]: copy.deepcopy(records[c["id"]]["graph"]) for c in self.courses() if c["id"] in records}

    def get(self, *, course_id=None, name=None):
        course = self.resolve(name=name, course_id=course_id)
        with self.lock:
            record = self._records(course["id"], review_only=True).get(course["id"])
            return copy.deepcopy(record["graph"]) if record else {"nodes": [], "links": []}

    def _save(self, course, graph, model, previous_revision, remember=True, native_before=None, update_model=True):
        course = self.resolve(course_id=course.get("id"), name=course.get("name"))
        validate_course_graph(graph, course["id"])
        if native_before is not None:
            validate_course_graph(native_before, course["id"])
        revision = previous_revision + 1
        graph = clean_snapshot_graph(graph)
        model = clean_snapshot_graph(model)
        graph["review_revision"] = revision
        graph["node_count"] = len(graph.get("nodes", []))
        graph["link_count"] = len(graph.get("links", []))
        if self.config.get("DATABASE_MODE") == "neo4j":
            driver = self._driver()
            try:
                with driver.session() as session:
                    with session.begin_transaction() as tx:
                        # Take the course write lock before checking revision and changing entities.
                        row = tx.run("MERGE (c:Course {id:$id}) ON CREATE SET c.course_id=$id, c.course_name=$name "
                            "SET c.review_revision=coalesce(c.review_revision,0) "
                            "RETURN c.review_revision AS revision,c.review_graph_json AS review, "
                            "c.model_graph_json AS model,c.native_graph_version AS native_version, "
                            "c.undo_revision AS undo_revision,c.previous_review_graph_json AS previous",
                            id=course["id"], name=course["name"]).single()
                        if row["revision"] != previous_revision:
                            raise GraphError("图谱已被其他操作更新，请重新加载后再保存", 409)
                        old = native_before if native_before is not None else decode_snapshot(row["review"])
                        baseline = old if row["native_version"] else decode_snapshot(row["model"] or row["review"])
                        # Restoring a snapshot reconciles all relationships, including physical deletions.
                        apply_delta(tx, baseline, graph, course["id"], force=not row["native_version"] or not remember)
                        parent = row.get("undo_revision")
                        if parent is None:
                            parent = previous_revision if row.get("previous") else 0
                        if remember:
                            undo_revision = revision
                        else:
                            history = tx.run("MATCH (h:GraphRevision {owner_course_id:$id, revision:$revision}) RETURN h.previous_undo_revision AS parent",
                                id=course["id"], revision=parent).single() if parent else None
                            undo_revision = (history.get("parent") or 0) if history else 0
                        updated = datetime.now(timezone.utc).isoformat()
                        # Convert a large legacy model envelope once, without changing its content.
                        store_model = update_model or not (row["model"] and '"_graph_encoding"' in row["model"][:80])
                        model_envelope = model if update_model else decode_snapshot(row["model"] or row["review"])
                        # Durable history lives outside the course graph and can restore physical deletions.
                        audit = base64.b64encode(zlib.compress(json.dumps(old, ensure_ascii=False).encode('utf-8'))).decode('ascii')
                        tx.run("CREATE (h:GraphRevision {owner_course_id:$id, revision:$revision, "
                               "created_at:$updated, before_json_zlib:$before, previous_undo_revision:$parent, undoable:$undoable})",
                               id=course["id"], revision=revision, updated=updated, before=audit, parent=parent, undoable=remember).consume()
                        tx.run("MATCH (c:Course {id:$id}) SET c.previous_review_graph_json=$previous, "
                               "c.review_graph_json=$review, c.review_revision=$revision, "
                               "c.native_graph_version=1,c.review_updated_at=$updated,c.undo_revision=$undo_revision "
                               "FOREACH (_ IN CASE WHEN $update_model THEN [1] ELSE [] END | SET c.model_graph_json=$model)",
                               id=course["id"], previous=encode_snapshot(old) if remember else None,
                               review=encode_snapshot(graph),
                               model=encode_snapshot(model_envelope) if store_model else None,
                               update_model=store_model, revision=revision, updated=updated, undo_revision=undo_revision).consume()
                        tx.commit()
            except GraphError:
                raise
            except Exception:
                raise GraphError("图谱未保存成功，请检查数据库连接后重试", 503) from None
            finally:
                driver.close()
        self._map_metadata_cache[course["id"]] = {"revision": revision,
            "metadata": {k:copy.deepcopy(v) for k,v in graph.items() if k not in {"nodes", "links"}}}
        # Local copies are a cache/backup in Neo4j mode, never a silent fallback.
        self.cache_warning = None
        try:
            records = json.loads(self.records_path.read_text(encoding="utf-8")) if self.records_path.exists() else {}
            cached = records.get(course["id"], {})
            stack = cached.get("undo_stack")
            if stack is None:
                stack = [cached["previous_graph"]] if cached.get("previous_graph") is not None else []
            if remember and cached.get("graph") is not None:
                stack.append(cached["graph"])
            elif not remember and stack:
                stack.pop()
            previous = stack[-1] if stack else None
            records[course["id"]] = {"graph": graph, "model": model, "revision": revision, "previous_graph": previous, "undo_stack": stack}
            atomic_json(self.records_path, records)
            atomic_json(self.path, {c["name"]: records[c["id"]]["graph"] for c in self.courses() if c["id"] in records})
        except (OSError, ValueError):
            if self.config.get("DATABASE_MODE") != "neo4j":
                raise GraphError("本地图谱未保存成功，请检查运行目录权限", 503) from None
            self.cache_warning = "云端图谱已保存，本地缓存更新失败；请修复运行目录权限，勿重复付费提取。"
        return graph

    def mutate(self, operation, *arguments, expected_revision=None):
        with self.lock:
            payload = arguments[0] if arguments and isinstance(arguments[0], dict) else {}
            name = payload.get("course_name") or (arguments[1] if len(arguments) > 1 else None)
            selected = self.resolve(name=name)
            records = self._records(selected["id"])
            graphs = {selected["name"]: copy.deepcopy(records[selected["id"]]["graph"])} if selected["id"] in records else {}
            output, result = operation(graphs, *arguments)
            course = self.resolve(name=result.get("course_name"), course_id=selected["id"])
            record = records.get(course["id"], {"revision": 0, "model": {"nodes": [], "links": []}})
            if expected_revision is not None and record["revision"] != expected_revision:
                raise GraphError("图谱已被其他页面更新，请重新加载后再修改。", 409)
            graph = self._save(course, output[course["name"]], record["model"], record["revision"], native_before=record.get("graph"), update_model=False)
            result["review_revision"] = graph["review_revision"]
            if self.cache_warning:
                result["warnings"] = [self.cache_warning]
            self._sync_vectors(course, graph, result)
            return result

    def _sync_vectors(self, course, graph, result):
        if not self.config.get("ENABLE_VECTOR_RETRIEVAL"):
            result["vector_sync"] = {"status": "disabled"}
            return
        try:
            from .vector_service import sync_graph
            result["vector_sync"] = {"status": "synced", **sync_graph(self.config, graph, course["id"])}
        except Exception as error:
            logging.getLogger(__name__).warning("Vector sync pending: %s: %s", type(error).__name__, str(error)[:250])
            result["vector_sync"] = {"status": "pending"}
            result.setdefault("warnings", []).append("Neo4j已保存；向量同步暂未完成，下次问答前会重试同步。")

    def undo(self, course, expected_revision):
        course = self.resolve(course_id=course.get("id"), name=course.get("name"))
        with self.lock:
            record = self._records(course["id"]).get(course["id"])
            if not record or record["revision"] != expected_revision:
                raise GraphError("图谱版本已变化，请重新加载。", 409)
            previous = record.get("previous_graph")
            if previous is None:
                raise GraphError("没有可撤销的修改。", 409)
            graph = self._save(course, previous, record["model"], record["revision"], remember=False, native_before=record["graph"], update_model=False)
            result = {"success": True, "review_revision": graph["review_revision"]}
            self._sync_vectors(course, graph, result)
            return result

    def publish(self, job, result):
        course = self.resolve(course_id=job["course_id"], name=job["course_name"])
        model = clean_snapshot_graph(result["graph"])
        if not isinstance(model, dict) or not isinstance(model.get("nodes"), list) or not isinstance(model.get("links"), list):
            raise GraphError("图谱格式不完整，未发布", 409)
        for node in model["nodes"]:
            if "Course" in node.get("labels", []) or node.get("type") == "course":
                # A native export must not recursively embed earlier snapshots.
                for field in ("review_graph_json", "model_graph_json", "previous_review_graph_json", "review_revision", "review_updated_at"):
                    node.get("properties", {}).pop(field, None)
            identity = node.get("course_id") or node.get("properties", {}).get("course_id")
            if identity and identity != course["id"]:
                raise GraphError("图谱含其他课程的节点，未发布", 409)
        with self.lock:
            record = self._records(course["id"]).get(course["id"])
            graph = merge_review_graph(record["model"], record.get("stored_graph", record["graph"]), model) if record else model
            saved = self._save(course, graph, model, record["revision"] if record else 0, native_before=model)
        return {"graph_url": "/knowledge-graph/?" + urlencode({"course_id": course["id"], "course_name": course["name"]}),
                "review_revision": saved["review_revision"], "database_mode": self.config.get("DATABASE_MODE"),
                "api_budget": result.get("budget"), "warnings": [self.cache_warning] if self.cache_warning else []}


def repository():
    return current_app.extensions["graph_repository"]
