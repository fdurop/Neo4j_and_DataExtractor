from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services.qa import retrieve


class SourceTests(unittest.TestCase):
    def test_each_material_keeps_its_own_job_source(self):
        graph = {"integration": {"source_url": "/api/jobs/latest/source"}, "nodes": [
            {"id": "x", "type": "page", "name": "反馈", "description": "反馈测量", "properties": {"job_id": "original", "page_id": "p", "page_no": 3}}]}
        sources = retrieve(graph, "反馈测量")
        self.assertEqual(sources[0]["source_url"], "/api/jobs/original/source")
        self.assertEqual(sources[0]["page_no"], 3)

    def test_teacher_corrected_description_is_used_instead_of_old_generated_text(self):
        graph = {"nodes": [{"id": "x", "type": "figure", "name": "反馈图", "description": "反馈测量的教师修正文案",
                "properties": {"origin": "teacher_edit", "source_text": "原模型误读", "page_id": "p"}}]}
        sources = retrieve(graph, "反馈测量")
        self.assertIn("教师修正文案", sources[0]["text"])
        self.assertNotIn("原模型误读", sources[0]["text"])

    def test_material_page_number_and_original_job_are_resolved_from_page(self):
        graph = {"integration": {"source_url": "/api/jobs/latest/source"}, "nodes": [
            {"id": "page-storage-id", "type": "page", "properties": {"page_id": "page-source-id", "page_no": 7,
                "document_id": "document-original", "job_id": "job-original", "course_id": "course-a"}},
            {"id": "material", "type": "figure", "name": "反馈测量", "description": "反馈测量示意图",
                "properties": {"source_page_id": "page-source-id", "course_id": "course-a"}}]}
        source = retrieve(graph, "反馈测量")[0]
        self.assertEqual(source["page_no"], 7)
        self.assertEqual(source["document_id"], "document-original")
        self.assertEqual(source["job_id"], "job-original")
        self.assertEqual(source["source_url"], "/api/jobs/job-original/source")

    def test_all_material_kinds_resolve_page_id_or_source_page_id(self):
        nodes = [{"id": "p", "type": "page", "properties": {"page_no": 4, "document_id": "d", "job_id": "j"}}]
        for index, kind in enumerate(("text_chunk", "figure", "table", "formula")):
            nodes.append({"id": kind, "type": kind, "description": "反馈测量",
                          "properties": {"page_id" if index % 2 else "source_page_id": "p"}})
        sources = retrieve({"nodes": nodes}, "反馈测量")
        self.assertEqual(len(sources), 4)
        self.assertEqual({source["page_no"] for source in sources}, {4})
        self.assertEqual({source["source_url"] for source in sources}, {"/api/jobs/j/source"})

    def test_same_legacy_page_id_is_scoped_by_document_and_job(self):
        nodes = []
        for index, job in enumerate(("older", "newer"), 1):
            nodes.extend([
                {"id": "page-" + job, "type": "page", "properties": {"page_id": "shared", "page_no": index,
                    "document_id": "doc-" + job, "job_id": job}},
                {"id": "material-" + job, "type": "figure", "description": "反馈测量",
                    "properties": {"source_page_id": "shared", "document_id": "doc-" + job, "job_id": job}},
            ])
        sources = retrieve({"nodes": nodes}, "反馈测量")
        self.assertEqual({source["job_id"]: source["page_no"] for source in sources}, {"older": 1, "newer": 2})

    def test_conflicting_or_ambiguous_page_does_not_supply_a_guessed_page(self):
        graph = {"integration": {"source_url": "/api/jobs/latest/source"}, "nodes": [
            {"id": "p1", "type": "page", "properties": {"page_id": "shared", "page_no": 2, "job_id": "wrong"}},
            {"id": "p2", "type": "page", "properties": {"page_id": "shared", "page_no": 9, "job_id": "other"}},
            {"id": "ambiguous", "type": "table", "description": "反馈测量", "properties": {"source_page_id": "shared"}},
            {"id": "conflicting", "type": "figure", "description": "反馈测量", "properties": {"source_page_id": "shared", "job_id": "original"}},
        ]}
        sources = {source["node_id"]: source for source in retrieve(graph, "反馈测量")}
        self.assertIsNone(sources["ambiguous"]["page_no"])
        self.assertIsNone(sources["ambiguous"]["source_url"])
        self.assertIsNone(sources["conflicting"]["page_no"])
        self.assertEqual(sources["conflicting"]["source_url"], "/api/jobs/original/source")

    def test_unknown_page_stays_unknown_and_boolean_is_not_a_page_number(self):
        nodes = [
            {"id": "unknown", "type": "figure", "description": "反馈测量", "properties": {"source_page_id": "page_999"}},
            {"id": "boolean-page", "type": "page", "description": "反馈测量", "properties": {"page_no": True}},
        ]
        self.assertTrue(all(source["page_no"] is None for source in retrieve({"nodes": nodes}, "反馈测量")))


if __name__ == "__main__":
    unittest.main()
