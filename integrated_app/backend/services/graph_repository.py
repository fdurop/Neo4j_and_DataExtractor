"""Authoritative reviewed graph snapshots with atomic Neo4j revisions."""
import copy
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
from neo4j_runtime import driver_for
from .graph_merge import merge_review_graph


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class GraphRepository:
    def __init__(self, config):
        self.config = dict(config)
        self.path = Path(config["KG_GRAPH_PATH"])
        self.records_path = self.path.with_name("graph_records.json")
        self.lock = RLock()
        self.cache_warning = None

    def courses(self):
        path = Path(self.config["APP_DATA_DIR"]) / "courses.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def resolve(self, name=None, course_id=None):
        candidates = [c for c in self.courses() if not c.get("archived")
                      and (not course_id or c.get("id") == course_id or c.get("course_id") == course_id)
                      and (not name or c.get("name") == name)]
        if (not name and not course_id) or len(candidates) != 1:
            raise GraphError("课程不存在或课程标识不匹配", 404)
        return candidates[0]

    def _driver(self):
        c = self.config
        return driver_for(c["NEO4J_URI"], c["NEO4J_USER"], c["NEO4J_PASSWORD"],
                          c.get("NEO4J_DATABASE"), c.get("NEO4J_CA_BUNDLE"))

    def _records(self):
        if self.config.get("DATABASE_MODE") != "neo4j":
            return json.loads(self.records_path.read_text(encoding="utf-8")) if self.records_path.exists() else {}
        courses = self.courses()
        ids = [c["id"] for c in courses if not c.get("archived")]
        if not ids:
            return {}
        driver = self._driver()
        try:
            with driver.session(default_access_mode="READ") as session:
                rows = session.run("MATCH (c:Course) WHERE c.id IN $ids RETURN c.id AS id, "
                                   "c.review_graph_json AS review, c.model_graph_json AS model, "
                                   "coalesce(c.review_revision,0) AS revision, c.previous_review_graph_json AS previous", ids=ids)
                return {r["id"]: {"graph": json.loads(r["review"]), "model": json.loads(r["model"] or r["review"]),
                                  "revision": r["revision"], "previous_graph": json.loads(r["previous"]) if r.get("previous") else None} for r in rows if r["review"]}
        except Exception:
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
            record = self._records().get(course["id"])
            return copy.deepcopy(record["graph"]) if record else {"nodes": [], "links": []}

    def _save(self, course, graph, model, previous_revision, remember=True):
        revision = previous_revision + 1
        graph = copy.deepcopy(graph)
        graph["review_revision"] = revision
        graph["node_count"] = len(graph.get("nodes", []))
        graph["link_count"] = len(graph.get("links", []))
        if self.config.get("DATABASE_MODE") == "neo4j":
            driver = self._driver()
            try:
                with driver.session() as session:
                    with session.begin_transaction() as tx:
                        row = tx.run("MERGE (c:Course {id:$id}) ON CREATE SET c.course_id=$id, c.course_name=$name "
                                     "WITH c WHERE coalesce(c.review_revision,0)=$expected "
                                     "SET c.previous_review_graph_json=CASE WHEN $remember THEN c.review_graph_json ELSE null END "
                                     "SET c.review_graph_json=$review, c.model_graph_json=$model, "
                                     "c.review_revision=$revision, c.review_updated_at=$updated "
                                     "RETURN c.review_revision AS revision", id=course["id"], name=course["name"],
                                     expected=previous_revision, review=json.dumps(graph, ensure_ascii=False, allow_nan=False),
                                     model=json.dumps(model, ensure_ascii=False, allow_nan=False), revision=revision,
                                     updated=datetime.now(timezone.utc).isoformat(), remember=remember).single()
                        if row is None:
                            raise GraphError("图谱已被其他操作更新，请重新加载后再保存", 409)
                        tx.commit()
            except GraphError:
                raise
            except Exception:
                raise GraphError("图谱未保存成功，请检查数据库连接后重试", 503) from None
            finally:
                driver.close()
        # Local copies are a cache/backup in Neo4j mode, never a silent fallback.
        self.cache_warning = None
        try:
            records = json.loads(self.records_path.read_text(encoding="utf-8")) if self.records_path.exists() else {}
            previous = records.get(course["id"], {}).get("graph") if remember else None
            records[course["id"]] = {"graph": graph, "model": model, "revision": revision, "previous_graph": previous}
            atomic_json(self.records_path, records)
            atomic_json(self.path, {c["name"]: records[c["id"]]["graph"] for c in self.courses() if c["id"] in records})
        except (OSError, ValueError):
            if self.config.get("DATABASE_MODE") != "neo4j":
                raise GraphError("本地图谱未保存成功，请检查运行目录权限", 503) from None
            self.cache_warning = "云端图谱已保存，本地缓存更新失败；请修复运行目录权限，勿重复付费提取。"
        return graph

    def mutate(self, operation, *arguments, expected_revision=None):
        with self.lock:
            records = self._records()
            graphs = {c["name"]: copy.deepcopy(records[c["id"]]["graph"]) for c in self.courses() if c["id"] in records}
            output, result = operation(graphs, *arguments)
            course = self.resolve(name=result.get("course_name"))
            record = records.get(course["id"], {"revision": 0, "model": {"nodes": [], "links": []}})
            if expected_revision is not None and record["revision"] != expected_revision:
                raise GraphError("图谱已被其他页面更新，请重新加载后再修改。", 409)
            graph = self._save(course, output[course["name"]], record["model"], record["revision"])
            result["review_revision"] = graph["review_revision"]
            if self.cache_warning:
                result["warnings"] = [self.cache_warning]
            return result

    def undo(self, course, expected_revision):
        with self.lock:
            record = self._records().get(course["id"])
            if not record or record["revision"] != expected_revision:
                raise GraphError("图谱版本已变化，请重新加载。", 409)
            previous = record.get("previous_graph")
            if previous is None:
                raise GraphError("没有可撤销的修改。", 409)
            graph = self._save(course, previous, record["model"], record["revision"], remember=False)
            return {"success": True, "review_revision": graph["review_revision"]}

    def publish(self, job, result):
        course = self.resolve(course_id=job["course_id"], name=job["course_name"])
        model = copy.deepcopy(result["graph"])
        if not isinstance(model, dict) or not isinstance(model.get("nodes"), list) or not isinstance(model.get("links"), list):
            raise GraphError("图谱格式不完整，未发布", 409)
        for node in model["nodes"]:
            if "Course" in node.get("labels", []) or node.get("type") == "course":
                # A native export must not recursively embed earlier snapshots.
                for field in ("review_graph_json", "model_graph_json", "review_revision", "review_updated_at"):
                    node.get("properties", {}).pop(field, None)
            identity = node.get("course_id") or node.get("properties", {}).get("course_id")
            if identity and identity != course["id"]:
                raise GraphError("图谱含其他课程的节点，未发布", 409)
        with self.lock:
            record = self._records().get(course["id"])
            graph = merge_review_graph(record["model"], record["graph"], model) if record else model
            saved = self._save(course, graph, model, record["revision"] if record else 0)
        return {"graph_url": "/knowledge-graph/?" + urlencode({"course_id": course["id"], "course_name": course["name"]}),
                "review_revision": saved["review_revision"], "database_mode": self.config.get("DATABASE_MODE"),
                "api_budget": result.get("budget"), "warnings": [self.cache_warning] if self.cache_warning else []}


def repository():
    return current_app.extensions["graph_repository"]
