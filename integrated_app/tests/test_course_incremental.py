"""Multiple-PDF provenance and review retention without models or Neo4j."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engines" / "graph"))
from course_incremental import concept_provenance, course_audit_records
from backend.services.graph_repository import GraphRepository
from backend.routes import graph_contract


def sources(document):
    return {"evidence_node_ids": [document + "-material"],
            "source_page_ids": [document + "-page"],
            "source_document_ids": [document],
            "page_observations_json": json.dumps([
                {"page_id": document + "-page", "document_id": document, "description": "Source observation"}
            ])}


class IncrementalCourseTests(unittest.TestCase):
    def test_second_pdf_accumulates_sources_without_changing_teacher_content(self):
        teacher = {**sources("pdf-a"), "description": "Teacher correction", "review_status": "approved",
                   "teaching_role": "learning", "role_review_status": "approved"}
        incoming = {**sources("pdf-b"), "description": "Different model description", "review_status": "pending",
                    "teaching_role": "detail", "role_review_status": "pending"}
        before = copy.deepcopy((teacher, incoming))
        patch = concept_provenance(teacher, incoming)
        combined = {**teacher, **patch}
        for field in ("description", "review_status", "teaching_role", "role_review_status"):
            self.assertNotIn(field, patch, "Provenance updates cannot bypass the content/role lock")
            self.assertEqual(combined[field], teacher[field])
        self.assertEqual(combined["evidence_node_ids"], ["pdf-a-material", "pdf-b-material"])
        self.assertEqual(combined["source_page_ids"], ["pdf-a-page", "pdf-b-page"])
        self.assertEqual(combined["source_document_ids"], ["pdf-a", "pdf-b"])
        self.assertEqual(combined["evidence_count"], 2)
        self.assertEqual(len(json.loads(combined["page_observations_json"])), 2)
        self.assertEqual((teacher, incoming), before)

    def test_repeated_document_payload_is_idempotent_and_rejected_status_is_retained(self):
        previous = {**sources("pdf-a"), "description": "Rejected interpretation", "review_status": "rejected"}
        first = {**previous, **concept_provenance(previous, sources("pdf-b"))}
        repeated = {**first, **concept_provenance(first, sources("pdf-b"))}
        self.assertEqual(first, repeated)
        self.assertEqual(repeated["review_status"], "rejected")
        self.assertEqual(repeated["description"], "Rejected interpretation")
        self.assertEqual(repeated["evidence_count"], 2)

    def test_course_audit_keeps_both_documents_and_deduplicates_exact_records(self):
        old = [{"id": "issue-a", "document_id": "pdf-a", "reason": "Needs review"}]
        new = [{"id": "issue-b", "document_id": "pdf-b", "reason": "Another question"}, old[0]]
        before = copy.deepcopy((old, new))
        merged = course_audit_records(json.dumps(old), new)
        self.assertEqual([item["id"] for item in merged], ["issue-a", "issue-b"])
        self.assertEqual(course_audit_records(merged, new), merged)
        self.assertEqual((old, new), before)

    def test_same_candidate_merges_evidence_but_keeps_reviewed_text_and_other_candidates(self):
        old = [{"source": "shared", "target": "target", "relation": "IS_A", "description": "Teacher wording",
                "review_status": "approved", "evidence_node_ids": ["pdf-a-material"], "source_page_ids": ["pdf-a-page"]}]
        new = [{"source": "shared", "target": "target", "relation": "IS_A", "description": "New model wording",
                "review_status": "pending", "evidence_node_ids": ["pdf-b-material"], "source_page_ids": ["pdf-b-page"]},
               {"source": "other", "target": "target", "relation": "PART_OF", "review_status": "pending"}]
        before = copy.deepcopy((old, new))
        merged = course_audit_records(old, new, candidates=True)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["description"], "Teacher wording")
        self.assertEqual(merged[0]["review_status"], "approved")
        self.assertEqual(merged[0]["evidence_node_ids"], ["pdf-a-material", "pdf-b-material"])
        self.assertEqual(merged[0]["source_page_ids"], ["pdf-a-page", "pdf-b-page"])
        self.assertEqual(merged[0]["evidence_count"], 2)
        self.assertEqual(merged[1]["source"], "other")
        self.assertEqual(course_audit_records(merged, new, candidates=True), merged)
        self.assertEqual((old, new), before)

    def test_corrupt_stored_sources_fail_visibly_instead_of_resetting_prior_documents(self):
        with self.assertRaises(ValueError):
            concept_provenance({"source_page_ids": "not-an-array"}, sources("pdf-b"))
        with self.assertRaises(ValueError):
            concept_provenance({"page_observations_json": "invalid-json"}, sources("pdf-b"))
        with self.assertRaises(ValueError):
            course_audit_records('{"not": "a list"}', [])

    def test_two_document_publish_keeps_materials_other_concepts_and_teacher_edits(self):
        # This is a real local repository publish/review/publish cycle. The
        # cumulative native-export equivalent retains both PDFs' actual nodes.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "courses.json").write_text(json.dumps([
                {"id": "course-a", "name": "Course A"}, {"id": "course-b", "name": "Course B"}
            ]), encoding="utf-8")
            repo = GraphRepository({"APP_DATA_DIR": str(directory), "KG_GRAPH_PATH": str(directory / "graphs.json"),
                                    "DATABASE_MODE": "local"})
            job = {"course_id": "course-a", "course_name": "Course A"}
            properties = {"course_id": "course-a", "review_status": "pending", **sources("pdf-a")}
            first = {"nodes": [
                {"id": "shared-concept", "name": "Shared concept", "type": "concept", "description": "Initial model wording",
                 "properties": properties},
                {"id": "pdf-a", "type": "document", "properties": {"course_id": "course-a"}},
                {"id": "pdf-a-page", "type": "page", "properties": {"course_id": "course-a", "document_id": "pdf-a"}},
                {"id": "pdf-a-material", "type": "figure", "properties": {"course_id": "course-a", "source_page_id": "pdf-a-page"}},
            ], "links": [{"source": "shared-concept", "target": "pdf-a-material", "relation": "HAS_EVIDENCE"}]}
            repo.publish(job, {"graph": first})
            repo.mutate(graph_contract.edit_node, {"course_name": "Course A", "id": "shared-concept",
                "description": "Teacher's corrected explanation", "review_status": "approved"})
            second = copy.deepcopy(first)
            shared = second["nodes"][0]
            shared["description"] = "Second PDF model wording"
            shared["properties"].update(concept_provenance(properties, sources("pdf-b")))
            second["nodes"].extend([
                {"id": "pdf-b", "type": "document", "properties": {"course_id": "course-a"}},
                {"id": "pdf-b-page", "type": "page", "properties": {"course_id": "course-a", "document_id": "pdf-b"}},
                {"id": "pdf-b-material", "type": "figure", "properties": {"course_id": "course-a", "source_page_id": "pdf-b-page"}},
                {"id": "different-concept", "name": "Different concept", "type": "concept",
                 "properties": {"course_id": "course-a", "review_status": "pending", **sources("pdf-b")}},
            ])
            second["links"].append({"source": "shared-concept", "target": "pdf-b-material", "relation": "HAS_EVIDENCE"})
            repo.publish(job, {"graph": second})
            published = repo.get(course_id="course-a")
            nodes = {node["id"]: node for node in published["nodes"]}
            self.assertEqual(set(nodes), {"shared-concept", "different-concept", "pdf-a", "pdf-b",
                                         "pdf-a-page", "pdf-b-page", "pdf-a-material", "pdf-b-material"})
            self.assertEqual(nodes["shared-concept"]["description"], "Teacher's corrected explanation")
            self.assertEqual(nodes["shared-concept"]["properties"]["review_status"], "approved")
            self.assertEqual(nodes["shared-concept"]["properties"]["source_document_ids"], ["pdf-a", "pdf-b"])
            self.assertEqual(nodes["shared-concept"]["properties"]["evidence_count"], 2)
            self.assertEqual({link["target"] for link in published["links"]}, {"pdf-a-material", "pdf-b-material"})
            self.assertEqual(repo.get(course_id="course-b")["nodes"], [])


if __name__ == "__main__":
    unittest.main()
