"""Local lifecycle integration checks; real Flask/SQLite/JSON, no provider calls."""
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app import create_app
from backend.models.db import db
from backend.services.jobs import JobCancelled
import fitz


def pdf_bytes():
    with fitz.open() as document:
        document.new_page().insert_text((72, 72), "Integration source document")
        return document.tobytes()


def simple_graph(job):
    return {"nodes": [{"id": job["course_id"] + ":concept", "name": "Shared concept", "type": "concept"}], "links": []}


class JobsApiTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "tests/.tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.block = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden in lifecycle tests"))
        self.block.start()
        self.apps = []
        self.app = self.make_app()
        self.client = self.app.test_client()
        self.password = "test-only-long-password"
        response = self.client.post("/register", json={"username": "teacher", "email": "teacher@example.invalid", "password": self.password, "role": "teacher"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.post("/login", json={"username": "teacher", "password": self.password}).status_code, 200)
        self.course = self.client.post("/api/courses", json={"name": "Generic course"}).get_json()

    def make_app(self, **extra):
        config = {"APP_DATA_DIR": self.temp.name, "SECRET_KEY": "isolated-session-key", "AUTO_START_JOBS": False, "TESTING": True}
        config.update(extra)
        app = create_app(config)
        self.apps.append(app)
        return app

    def tearDown(self):
        for app in self.apps:
            app.extensions["job_service"].shutdown()
            with app.app_context():
                db.session.remove()
                db.engine.dispose()
        self.block.stop()
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to((ROOT / "tests/.tmp").resolve()))
        self.temp.cleanup()

    def upload(self, *, client=None, course=None, filename="lecture.pdf", body=None):
        client, course = client or self.client, course or self.course
        return client.post("/api/upload", data={"course_id": course["id"], "course_name": course["name"],
                           "file": (io.BytesIO(pdf_bytes() if body is None else body), filename)}, content_type="multipart/form-data")

    def wait(self, job_id):
        service = self.app.extensions["job_service"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            record = service.get(job_id)
            if record["status"] in {"failed", "succeeded", "cancelled"}:
                return record
            time.sleep(0.01)
        self.fail("Worker did not reach a terminal state")

    def test_import_and_factory_do_not_load_engine_or_change_working_directory(self):
        import os
        before = os.getcwd()
        self.assertNotIn("multimodal_preprocessor", sys.modules)
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(os.getcwd(), before)

    def test_unset_config_secret_preserves_generated_sessions_across_restart(self):
        app = self.make_app(SECRET_KEY=None)
        self.assertTrue(app.secret_key)
        client = app.test_client()
        response = client.post("/login", json={"username": "teacher", "password": self.password})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(client.get("/api/session").get_json()["authenticated"])
        restarted = self.make_app(SECRET_KEY=None)
        self.assertEqual(app.secret_key, restarted.secret_key)

    def test_cookie_session_and_teacher_permissions_are_real(self):
        guest = self.app.test_client()
        self.assertFalse(guest.get("/api/session").get_json()["authenticated"])
        self.assertEqual(guest.post("/api/courses", json={"name": "No auth"}).status_code, 401)
        guest.post("/register", json={"username": "student", "email": "student@example.invalid", "password": self.password})
        guest.post("/login", json={"username": "student", "password": self.password})
        self.assertEqual(guest.post("/api/courses", json={"name": "No permission"}).status_code, 403)
        self.assertEqual(self.client.get("/api/session").get_json()["user"]["role"], "teacher")

    def test_invalid_pdf_rejected_before_job_or_api_creation(self):
        response = self.upload(body=b"Not a PDF")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/jobs").get_json()["jobs"], [])

    def test_source_filename_cannot_escape_job_directory(self):
        self.assertEqual(self.upload(filename="../escape.pdf").status_code, 400)
        self.assertEqual(self.upload(filename="bad.txt").status_code, 400)

    def test_course_identity_mismatch_rejected(self):
        response = self.client.post("/api/upload", data={"course_id": self.course["id"], "course_name": "Wrong name",
                    "file": (io.BytesIO(pdf_bytes()), "source.pdf")}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_same_named_uploads_have_distinct_directories_and_filter_by_course(self):
        second = self.client.post("/api/courses", json={"name": "Another course"}).get_json()
        first_job = self.upload().get_json()["job"]
        second_job = self.upload(course=second).get_json()["job"]
        self.assertNotEqual(first_job["id"], second_job["id"])
        service = self.app.extensions["job_service"]
        self.assertNotEqual(service.get(first_job["id"], private=True)["input_path"], service.get(second_job["id"], private=True)["input_path"])
        filtered = self.client.get("/api/jobs", query_string={"course_id": self.course["id"]}).get_json()["jobs"]
        self.assertEqual([job["id"] for job in filtered], [first_job["id"]])

    def test_completed_job_includes_published_graph_and_100_percent(self):
        self.app.config["PIPELINE_RUNNER"] = lambda job, update, cancelled: {"graph": simple_graph(job), "artifacts": {}}
        identity = self.upload().get_json()["job_id"]
        self.app.extensions["job_service"].start(identity)
        completed = self.wait(identity)
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["progress"], 100)
        graph = self.client.get("/api/graph", query_string={"course_name": self.course["name"]}).get_json()
        self.assertEqual(graph["nodes"][0]["name"], "Shared concept")
        self.assertIn("course_id=" + self.course["id"], completed["graph_url"])

    def test_pipeline_failure_is_persistent_and_visible_after_reload(self):
        def fail(job, update, cancelled):
            update(stage="extracting", progress=15, message="Extracting")
            raise RuntimeError("controlled parse failure")
        self.app.config["PIPELINE_RUNNER"] = fail
        identity = self.upload().get_json()["job_id"]
        self.app.extensions["job_service"].start(identity)
        failed = self.wait(identity)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["progress"], 15)
        self.assertIn("controlled parse failure", failed["error"])
        states = self.client.get("/api/file-status", query_string={"course_id": self.course["id"]}).get_json()
        self.assertEqual(states[0]["status"], "failed")
        restarted = self.make_app()
        self.assertEqual(restarted.extensions["job_service"].get(identity)["status"], "failed")

    def test_publisher_failure_cannot_be_called_success(self):
        self.app.config["PIPELINE_RUNNER"] = lambda job, update, cancelled: {"graph": simple_graph(job)}
        def fail_publish(job, result):
            raise RuntimeError("controlled database publication failure")
        self.app.config["PIPELINE_PUBLISHER"] = fail_publish
        identity = self.upload().get_json()["job_id"]
        self.app.extensions["job_service"].start(identity)
        failed = self.wait(identity)
        self.assertEqual(failed["status"], "failed")
        self.assertLess(failed["progress"], 100)
        self.assertIn("publication", failed["error"])

    def test_restarting_queued_or_running_job_never_silently_retries_paid_work(self):
        identity = self.upload().get_json()["job_id"]
        restarted = self.make_app()
        record = restarted.extensions["job_service"].get(identity)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["stage"], "interrupted")

    def test_queued_cancellation_sends_no_worker_request(self):
        identity = self.upload().get_json()["job_id"]
        response = self.client.post("/api/jobs/" + identity + "/cancel")
        self.assertEqual(response.get_json()["job"]["status"], "cancelled")
        self.assertEqual(self.app.extensions["job_service"].futures, {})

    def test_running_cancellation_waits_for_worker_confirmation(self):
        entered = threading.Event()
        allow = threading.Event()
        def runner(job, update, cancelled):
            entered.set()
            allow.wait(timeout=3)
            if cancelled():
                raise JobCancelled("Cancelled at checkpoint")
            return {"graph": simple_graph(job)}
        self.app.config["PIPELINE_RUNNER"] = runner
        identity = self.upload().get_json()["job_id"]
        self.app.extensions["job_service"].start(identity)
        self.assertTrue(entered.wait(timeout=2))
        response = self.client.post("/api/jobs/" + identity + "/cancel")
        self.assertEqual(response.get_json()["job"]["status"], "cancel_requested")
        allow.set()
        self.assertEqual(self.wait(identity)["status"], "cancelled")

    def test_secrets_in_worker_exception_are_redacted(self):
        self.app.config["DASHSCOPE_API_KEY"] = "secret-value-used-only-in-test"
        def runner(job, update, cancelled):
            raise RuntimeError("provider error api_key=secret-value-used-only-in-test")
        self.app.config["PIPELINE_RUNNER"] = runner
        identity = self.upload().get_json()["job_id"]
        self.app.extensions["job_service"].start(identity)
        failed = self.wait(identity)
        self.assertNotIn("secret-value-used-only-in-test", failed["error"])

    def test_source_download_and_hide_do_not_delete_paid_results(self):
        identity = self.upload().get_json()["job_id"]
        self.client.post("/api/jobs/" + identity + "/cancel")
        response = self.client.get("/api/jobs/" + identity + "/source")
        self.assertEqual(response.status_code, 200)
        response.close()
        self.assertEqual(self.client.delete("/api/files/" + identity).status_code, 200)
        self.assertEqual(self.client.get("/api/files").get_json()["files"], [])
        self.assertTrue(Path(self.app.extensions["job_service"].get(identity, private=True)["input_path"]).exists())

    def test_missing_qa_configuration_is_not_a_successful_answer(self):
        def unavailable(question, course):
            raise RuntimeError("Not configured")
        self.app.config["QA_RUNNER"] = unavailable
        response = self.client.post("/api/qa/", json={"question": "Explain this concept", "course_id": self.course["id"]})
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["success"])


if __name__ == "__main__":
    unittest.main()
