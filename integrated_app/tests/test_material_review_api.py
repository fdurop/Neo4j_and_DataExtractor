"""Material review integration using real local graph persistence and assets."""
import copy
import io
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app import create_app
from backend.models.db import db
from backend.services.graph_repository import GraphRepository
from PIL import Image
import fitz


def source_pdf():
    with fitz.open() as document:
        for index in range(3):
            document.new_page().insert_text((72, 72), "Source page " + str(index + 1))
        return document.tobytes()


class MaterialReviewTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "tests/.tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.block = patch.object(socket.socket, "connect", side_effect=AssertionError("No external network"))
        self.block.start()
        self.app = create_app({"APP_DATA_DIR": self.temp.name, "SECRET_KEY": "review-test-session", "TESTING": True,
                               "AUTO_START_JOBS": False, "DATABASE_MODE": "local"})
        self.repo = GraphRepository(self.app.config)
        self.app.extensions["graph_repository"] = self.repo
        self.client = self.app.test_client()
        self.password = "test-review-only-password"
        self.client.post("/register", json={"username": "reviewer", "email": "reviewer@example.invalid", "password": self.password, "role": "teacher"})
        self.client.post("/login", json={"username": "reviewer", "password": self.password})
        self.course, self.job, self.graph = self.seed("Review course")

    def tearDown(self):
        self.app.extensions["job_service"].shutdown()
        with self.app.app_context():
            db.session.remove()
            db.engine.dispose()
        self.block.stop()
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to((ROOT / "tests/.tmp").resolve()))
        self.temp.cleanup()

    def seed(self, name):
        course = self.client.post("/api/courses", json={"name": name}).get_json()
        upload = self.client.post("/api/upload", data={"course_id": course["id"],
                   "file": (io.BytesIO(source_pdf()), "review.pdf")}, content_type="multipart/form-data").get_json()
        job = self.app.extensions["job_service"].get(upload["job_id"], private=True)
        assets = Path(job["job_dir"]) / "extraction/assets"
        assets.mkdir(parents=True)
        image = assets / "page_03.png"
        Image.new("RGB", (40, 30), "white").save(image)
        shared = {"course_id": course["id"], "job_id": job["id"]}
        graph = {"nodes": [
            {"id": "page-shared", "type": "page", "name": "Source page", "properties": {**shared, "page_id": "page-shared", "page_no": 3, "page_image_path": str(image)}},
            {"id": "figure-shared", "type": "figure", "name": "Illustration", "description": "Model description", "properties": {**shared, "source_page_id": "page-shared", "source_path": str(image), "review_status": "pending"}},
            {"id": "table-approved", "type": "table", "name": "Approved table", "properties": {**shared, "source_page_id": "page-shared", "review_status": "approved"}},
            {"id": "formula-shared", "type": "formula", "name": "Formula", "properties": {**shared, "source_page_id": "page-shared", "latex": "x = y", "review_status": "pending"}},
            {"id": "concept-shared", "type": "concept", "name": "Concept", "properties": {**shared, "review_status": "pending", "role_review_status": "pending"}},
        ], "links": [{"id": "relation-shared", "source": "figure-shared", "target": "concept-shared", "relation": "MENTIONS", "review_status": "pending", "properties": {"course_id": course["id"], "review_status": "pending", "evidence_ids": ["figure-shared"]}}]}
        self.repo.publish(job, {"graph": graph})
        return course, job, graph

    def test_teacher_list_contains_only_pending_materials_and_source_page(self):
        response = self.client.get("/api/repair/items", query_string={"course_id": self.course["id"]})
        self.assertEqual(response.status_code, 200)
        items = response.get_json()["items"]
        self.assertEqual({item["id"] for item in items}, {"figure-shared", "formula-shared"})
        figure = next(item for item in items if item["id"] == "figure-shared")
        self.assertEqual(figure["source_url"], "/api/jobs/" + self.job["id"] + "/source#page=3")
        self.assertEqual(figure["page_no"], 3)
        self.assertEqual(figure["image_url"], "/api/jobs/" + self.job["id"] + "/assets/page_03.png")
        self.assertEqual(next(item for item in items if item["id"] == "formula-shared")["description"], "x = y")

    def test_material_edit_persists_without_changing_concept_relation_or_evidence(self):
        before = self.repo.get(course_id=self.course["id"])
        response = self.client.put("/api/repair/items/figure-shared", json={"course_id": self.course["id"], "description": "Teacher corrected description"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["message"], "素材文案已更新，相关图谱关系仍需核对")
        after = self.repo.get(course_id=self.course["id"])
        material = next(node for node in after["nodes"] if node["id"] == "figure-shared")
        self.assertEqual(material["description"], "Teacher corrected description")
        self.assertEqual(material["properties"]["description"], "Teacher corrected description")
        self.assertEqual(material["properties"]["review_status"], "pending")
        self.assertEqual(material["properties"]["source_path"], self.graph["nodes"][1]["properties"]["source_path"])
        self.assertEqual(after["links"], before["links"])
        self.assertEqual(next(node for node in after["nodes"] if node["id"] == "concept-shared"),
                         next(node for node in before["nodes"] if node["id"] == "concept-shared"))

    def test_description_endpoint_cannot_approve_or_edit_concepts(self):
        response = self.client.put("/api/repair/items/figure-shared", json={"course_id": self.course["id"], "description": "Changed", "review_status": "approved"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.put("/api/repair/items/concept-shared", json={"course_id": self.course["id"], "description": "Changed"}).status_code, 404)

    def test_material_edit_requires_course_even_when_id_is_unique(self):
        before = self.repo.get(course_id=self.course["id"])
        response = self.client.put("/api/repair/items/figure-shared", json={"description": "No scope"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.repo.get(course_id=self.course["id"]), before)

    def test_material_edit_rejects_query_body_course_conflict(self):
        other, _, _ = self.seed("Another review course")
        before = self.repo.get(course_id=self.course["id"])
        response = self.client.put("/api/repair/items/figure-shared?course_id=" + other["id"],
            json={"course_id": self.course["id"], "description": "Conflicting scope"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.repo.get(course_id=self.course["id"]), before)

    def test_material_id_requires_course_when_ambiguous(self):
        other, _, _ = self.seed("Second review course")
        self.assertEqual(self.client.put("/api/repair/items/figure-shared", json={"description": "Ambiguous"}).status_code, 409)
        self.assertEqual(self.client.put("/api/repair/items/figure-shared", json={"course_id": other["id"], "description": "Only second course"}).status_code, 200)
        original = self.repo.get(course_id=self.course["id"])
        self.assertEqual(next(node for node in original["nodes"] if node["id"] == "figure-shared")["description"], "Model description")
        filtered = self.client.get("/api/repair/items", query_string={"course_id": other["id"]}).get_json()["items"]
        self.assertTrue(all(item["course_id"] == other["id"] for item in filtered))

    def test_material_review_requires_teacher_and_assets_require_login(self):
        guest = self.app.test_client()
        asset_url = "/api/jobs/" + self.job["id"] + "/assets/page_03.png"
        self.assertEqual(guest.get("/api/repair/items").status_code, 401)
        self.assertEqual(guest.get(asset_url).status_code, 401)
        guest.post("/register", json={"username": "student", "email": "student@example.invalid", "password": self.password})
        guest.post("/login", json={"username": "student", "password": self.password})
        self.assertEqual(guest.get("/api/repair/items").status_code, 403)
        self.assertEqual(guest.put("/api/repair/items/figure-shared", json={"course_id": self.course["id"], "description": "Attempt"}).status_code, 403)
        response = guest.get(asset_url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        response.close()

    def test_asset_endpoint_blocks_traversal_invalid_job_and_non_image_files(self):
        assets = Path(self.job["job_dir"]) / "extraction/assets"
        (assets / "not_image.png").write_text("not a PNG", encoding="utf-8")
        (assets / "script.svg").write_text("<svg></svg>", encoding="utf-8")
        Image.new("RGB", (10, 10)).save(Path(self.temp.name) / "outside.png")
        prefix = "/api/jobs/" + self.job["id"] + "/assets/"
        for suffix in ("not_image.png", "script.svg", "..%2f..%2f..%2foutside.png", "%00.png", "C:%5Coutside.png"):
            self.assertEqual(self.client.get(prefix + suffix).status_code, 404, suffix)
        self.assertEqual(self.client.get("/api/jobs/not-valid/assets/page_03.png").status_code, 404)

    def test_missing_asset_falls_back_to_source_without_local_path(self):
        graph = self.repo.get(course_id=self.course["id"])
        for node in graph["nodes"]:
            node.get("properties", {}).pop("page_image_path", None)
            node.get("properties", {}).pop("source_path", None)
        self.repo.publish(self.job, {"graph": graph})
        items = self.client.get("/api/repair/items", query_string={"course_id": self.course["id"]}).get_json()["items"]
        self.assertTrue(all("image_url" not in item for item in items))
        self.assertTrue(all(item["source_url"].endswith("#page=3") for item in items))
        self.assertNotIn(self.temp.name, str(items))

    def test_real_exporter_relative_image_paths_resolve_from_engine_workspace(self):
        graph = self.repo.get(course_id=self.course["id"])
        asset = Path(self.job["job_dir"]) / "extraction/assets/page_03.png"
        relative = os.path.relpath(asset, ROOT / "engines").replace(os.sep, "/")
        self.assertTrue(relative.startswith("../"))
        for node in graph["nodes"]:
            for field in ("source_path", "page_image_path"):
                if field in node.get("properties", {}):
                    node["properties"][field] = relative
        self.repo.publish(self.job, {"graph": graph})
        items = self.client.get("/api/repair/items", query_string={"course_id": self.course["id"]}).get_json()["items"]
        self.assertTrue(all(item.get("image_url", "").endswith("/assets/page_03.png") for item in items))

    def test_missing_material_image_can_use_its_page_in_exporter_relative_format(self):
        graph = self.repo.get(course_id=self.course["id"])
        asset = Path(self.job["job_dir"]) / "extraction/assets/page_03.png"
        relative = os.path.relpath(asset, ROOT / "engines").replace(os.sep, "/")
        for node in graph["nodes"]:
            data = node.get("properties", {})
            if node["type"] == "page":
                data["page_image_path"] = relative
            else:
                data.pop("source_path", None)
                data["enhanced_path"] = relative.replace("page_03.png", "missing_enhanced.png")
        self.repo.publish(self.job, {"graph": graph})
        items = self.client.get("/api/repair/items", query_string={"course_id": self.course["id"]}).get_json()["items"]
        self.assertTrue(all(item.get("image_url", "").endswith("/assets/page_03.png") for item in items))

    def test_relative_path_from_another_job_cannot_borrow_same_named_local_asset(self):
        other, other_job, _ = self.seed("Other job source")
        foreign_asset = Path(other_job["job_dir"]) / "extraction/assets/page_03.png"
        relative = os.path.relpath(foreign_asset, ROOT / "engines").replace(os.sep, "/")
        graph = self.repo.get(course_id=self.course["id"])
        for node in graph["nodes"]:
            data = node.get("properties", {})
            data.pop("page_image_path", None)
            data["source_path"] = relative
        self.repo.publish(self.job, {"graph": graph})
        items = self.client.get("/api/repair/items", query_string={"course_id": self.course["id"]}).get_json()["items"]
        self.assertTrue(all("image_url" not in item for item in items))
        self.assertTrue(all(item["source_url"].startswith("/api/jobs/" + self.job["id"] + "/") for item in items))

    def test_delete_is_not_exposed_and_keeps_evidence(self):
        response = self.client.delete("/api/repair/items/figure-shared")
        self.assertEqual(response.status_code, 405)
        graph = self.repo.get(course_id=self.course["id"])
        self.assertTrue(any(node["id"] == "figure-shared" for node in graph["nodes"]))


if __name__ == "__main__":
    unittest.main()
