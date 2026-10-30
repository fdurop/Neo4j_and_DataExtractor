import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services.graph_repository import GraphRepository
from backend.routes import graph_contract
from backend.app import create_app
from backend.models.db import db


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        (self.path / "courses.json").write_text(json.dumps([{"id": "a", "name": "课程A"}, {"id": "b", "name": "课程B"}]), encoding="utf-8")
        self.repo = GraphRepository({"APP_DATA_DIR": str(self.path), "KG_GRAPH_PATH": str(self.path / "graphs.json"), "DATABASE_MODE": "local"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_publish_edit_reload_and_model_refresh_preserve_review(self):
        model = {"nodes": [{"id": "x", "name": "概念", "type": "concept", "description": "模型说明",
                            "properties": {"course_id": "a", "review_status": "pending"}}], "links": []}
        job = {"course_id": "a", "course_name": "课程A"}
        self.repo.publish(job, {"graph": model})
        self.repo.mutate(graph_contract.edit_node, {"course_name": "课程A", "id": "x", "description": "教师说明", "review_status": "approved"})
        fresh = copy.deepcopy(model)
        fresh["nodes"][0]["description"] = "新模型说明"
        self.repo.publish(job, {"graph": fresh})
        second = GraphRepository(self.repo.config)
        node = second.get(course_id="a")["nodes"][0]
        self.assertEqual(node["description"], "教师说明")
        self.assertEqual(node["properties"]["review_status"], "approved")
        self.assertEqual(second.get(course_id="b")["nodes"], [])

    def test_course_identity_mismatch_is_rejected(self):
        with self.assertRaises(graph_contract.GraphError):
            self.repo.resolve(name="课程B", course_id="a")

    def test_published_course_cannot_recursively_embed_previous_snapshots(self):
        graph = {"nodes": [{"id": "a", "type": "course", "labels": ["Course"],
                           "properties": {"course_id": "a", "review_graph_json": "old snapshot", "model_graph_json": "old model"}}], "links": []}
        self.repo.publish({"course_id": "a", "course_name": "课程A"}, {"graph": graph})
        published = self.repo.get(course_id="a")
        self.assertNotIn("review_graph_json", published["nodes"][0]["properties"])
        self.assertNotIn("model_graph_json", published["nodes"][0]["properties"])
        self.assertIn("review_graph_json", graph["nodes"][0]["properties"])

    def test_failed_mutation_keeps_revision_and_cache(self):
        self.repo.publish({"course_id": "a", "course_name": "课程A"}, {"graph": {"nodes": [], "links": []}})
        original = self.repo.get(course_id="a")
        with self.assertRaises(graph_contract.GraphError):
            self.repo.mutate(graph_contract.edit_node, {"course_name": "课程A", "id": "missing", "name": "不能修改"})
        self.assertEqual(self.repo.get(course_id="a"), original)

    def test_query_course_identity_cannot_mutate_another_course_when_name_is_omitted(self):
        self.repo.publish({"course_id": "a", "course_name": "课程A"}, {"graph": {"nodes": [], "links": []}})
        self.repo.publish({"course_id": "b", "course_name": "课程B"}, {"graph": {"nodes": [{"id": "b-node", "name": "乙", "properties": {"course_id": "b"}}], "links": []}})
        app = create_app({**self.repo.config, "REQUIRE_AUTH": False, "TESTING": True})
        app.extensions["graph_repository"] = self.repo
        try:
            client = app.test_client()
            response = client.put("/api/graph/node?course_id=a", json={"id": "b-node", "name": "不能越过课程"})
            self.assertEqual(response.status_code, 404)
            self.assertEqual(self.repo.get(course_id="b")["nodes"][0]["name"], "乙")
        finally:
            app.extensions["job_service"].shutdown()
            with app.app_context():
                db.session.remove()
                db.engine.dispose()


if __name__ == "__main__":
    unittest.main()
