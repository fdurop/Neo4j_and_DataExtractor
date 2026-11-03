from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engines/extraction/src2"))
sys.path.insert(0, str(ROOT / "engines/graph"))
from utils.knowledge_exporter import KnowledgeExporter
from neo4j_runtime import driver_for, TrustCustomCAs
from backend.services.instance_lock import InstanceLock


class StorageTests(unittest.TestCase):
    def test_repeated_filename_different_jobs_have_distinct_material_identity(self):
        a = KnowledgeExporter(course_id="course-a", job_id="job-a")
        b = KnowledgeExporter(course_id="course-a", job_id="job-b")
        self.assertNotEqual(a.document_id("same.pdf"), b.document_id("same.pdf"))
        self.assertNotEqual(a.page_id("same.pdf", 0), b.page_id("same.pdf", 0))
        self.assertNotEqual(a.element_id("same.pdf", 0, "figure", 0), b.element_id("same.pdf", 0, "figure", 0))
        self.assertEqual(a.document_id("same.pdf"), a.document_id("same.pdf"))

    def test_cloud_connection_keeps_tls_hostname_verification_and_database(self):
        class Driver:
            def session(self, **kwargs):
                return kwargs
        with patch("neo4j_runtime.NativeGraphDatabase.driver", return_value=Driver()) as native:
            driver = driver_for("neo4j+s://example.invalid", "neo4j", "test-only", database="course-database")
            self.assertEqual(driver.session()["database"], "course-database")
            self.assertTrue(native.call_args.kwargs["encrypted"])
            self.assertIsInstance(native.call_args.kwargs["trusted_certificates"], TrustCustomCAs)

    def test_unverified_tls_or_remote_plaintext_is_rejected(self):
        for uri in ("neo4j+ssc://example.invalid", "bolt://example.invalid"):
            with self.assertRaises(ValueError):
                driver_for(uri, "neo4j", "test-only")

    def test_same_data_directory_cannot_start_two_service_instances(self):
        with tempfile.TemporaryDirectory() as path:
            with InstanceLock(path):
                with self.assertRaises(RuntimeError):
                    with InstanceLock(path):
                        pass
            with InstanceLock(path):
                pass


if __name__ == "__main__":
    unittest.main()
