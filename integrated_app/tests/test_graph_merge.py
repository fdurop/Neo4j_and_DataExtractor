"""Three-way review persistence independent of any model, Flask or database."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.graph_merge import merge_review_graph


def node(identity, name=None, **values):
    return {"id": identity, "name": name or identity, "review_status": "pending", **values}


def graph(nodes, links=None, **values):
    return {"course_id": "course_a", "nodes": nodes, "links": links or [], **values}


class ReviewGraphMergeTests(unittest.TestCase):
    def test_edited_fields_survive_while_untouched_evidence_updates(self):
        old = graph([node("a", description="old", sources=["page1"],
                          properties={"description": "old", "confidence": .4, "review_status": "pending"})])
        review, new = deepcopy(old), deepcopy(old)
        review["nodes"][0].update(name="教师名称", review_status="approved")
        review["nodes"][0]["properties"].update(description="教师依据", review_status="approved")
        new["nodes"][0].update(description="fresh", sources=["page1", "page2"])
        new["nodes"][0]["properties"].update(description="fresh", confidence=.9)
        merged = merge_review_graph(old, review, new)["nodes"][0]
        self.assertEqual(merged["name"], "教师名称")
        self.assertEqual(merged["description"], "fresh")
        self.assertEqual(merged["sources"], ["page1", "page2"])
        self.assertEqual(merged["properties"], {"description": "教师依据", "confidence": .9, "review_status": "approved"})
        self.assertEqual(merged["review_status"], "approved")

    def test_deleted_node_is_not_resurrected_and_incident_records_drop(self):
        edge = {"source": "a", "target": "b", "relation": "PART_OF"}
        old = graph([node("a"), node("b")], [edge], hierarchy_candidates=[deepcopy(edge)])
        review = deepcopy(old)
        review["nodes"] = [review["nodes"][0]]
        new = deepcopy(old)
        new["nodes"].append(node("c"))
        new["links"].append({"source": "a", "target": "c", "relation": "RELATED_TO"})
        merged = merge_review_graph(old, review, new)
        self.assertEqual([item["id"] for item in merged["nodes"]], ["a", "c"])
        self.assertEqual(merged["links"], [new["links"][1]])
        self.assertEqual(merged["hierarchy_candidates"], [])
        self.assertEqual((merged["node_count"], merged["link_count"]), (2, 1))

    def test_teacher_added_nodes_edges_and_audit_remain(self):
        old = graph([node("a")], review_items=[{"reason": "old-model"}])
        review = deepcopy(old)
        review["nodes"].append(node("manual", origin="teacher_edit", review_status="approved"))
        review["links"].append({"source": "a", "target": "manual", "relation": "RELATED_TO", "origin": "teacher_edit"})
        review["review_items"].append({"origin": "teacher_edit", "note": "保留核对记录"})
        review["teacher_notes"] = {"text": "老师补充"}
        new = graph([node("a"), node("new", description="新模型内容")], review_items=[{"reason": "new-model"}])
        merged = merge_review_graph(old, review, new)
        self.assertEqual({item["id"] for item in merged["nodes"]}, {"a", "manual", "new"})
        self.assertEqual(merged["nodes"][-1]["review_status"], "pending")
        self.assertEqual(merged["review_items"], [{"reason": "new-model"}, {"origin": "teacher_edit", "note": "保留核对记录"}])
        self.assertEqual(merged["teacher_notes"], {"text": "老师补充"})
        self.assertEqual(len(merged["links"]), 1)

    def test_edge_id_preserves_edits_when_model_endpoint_changes(self):
        old = graph([node("a"), node("b"), node("c")],
                    [{"id": "r", "source": "a", "target": "b", "relation": "RELATED_TO", "description": "old", "review_status": "pending"}])
        review, new = deepcopy(old), deepcopy(old)
        review["links"][0].update(description="教师解释", review_status="approved")
        new["links"][0].update(target="c", description="fresh")
        merged = merge_review_graph(old, review, new)["links"][0]
        self.assertEqual(merged["target"], "c")
        self.assertEqual(merged["description"], "教师解释")
        self.assertEqual(merged["review_status"], "approved")

    def test_relation_id_in_properties_is_a_stable_edge_identity(self):
        old = graph([node("a"), node("b")], [{"source": "a", "target": "b", "relation": "RELATED_TO",
                    "properties": {"relation_id": "stable", "review_status": "pending", "description": "old"}}])
        review, new = deepcopy(old), deepcopy(old)
        review["links"][0]["properties"]["review_status"] = "rejected"
        new["links"][0]["properties"]["description"] = "new"
        merged = merge_review_graph(old, review, new)["links"][0]
        self.assertEqual(merged["properties"]["review_status"], "rejected")
        self.assertEqual(merged["properties"]["description"], "new")

    def test_edge_without_id_uses_endpoints_and_relation(self):
        old = graph([node("a"), node("b")], [{"source": "a", "target": "b", "relation": "RELATED_TO", "description": "old"}])
        review, new = deepcopy(old), deepcopy(old)
        review["links"][0]["description"] = "review"
        new["links"][0]["description"] = "model"
        self.assertEqual(merge_review_graph(old, review, new)["links"][0]["description"], "review")

    def test_relation_change_without_id_is_review_addition_plus_old_deletion(self):
        old = graph([node("a"), node("b")], [{"source": "a", "target": "b", "relation": "RELATED_TO"}])
        review = deepcopy(old)
        review["links"][0].update(relation="PART_OF", review_status="approved", origin="teacher_edit")
        merged = merge_review_graph(old, review, deepcopy(old))
        self.assertEqual(merged["links"], review["links"])

    def test_deleted_edge_stays_deleted_and_new_pending_edge_is_retained(self):
        first = {"source": "a", "target": "b", "relation": "RELATED_TO"}
        second = {"source": "a", "target": "b", "relation": "PART_OF", "review_status": "pending"}
        old = graph([node("a"), node("b")], [first])
        review, new = deepcopy(old), deepcopy(old)
        review["links"] = []
        new["links"].append(second)
        self.assertEqual(merge_review_graph(old, review, new)["links"], [second])

    def test_new_model_does_not_remove_reviewed_entity(self):
        old = graph([node("a"), node("b")])
        review = deepcopy(old)
        review["nodes"][0]["review_status"] = "approved"
        merged = merge_review_graph(old, review, graph([]))
        self.assertEqual([item["id"] for item in merged["nodes"]], ["a"])

    def test_field_deletion_and_null_are_distinct(self):
        old = graph([node("a", description="old", nullable="old", properties={"note": "old", "confidence": .4})])
        review, new = deepcopy(old), deepcopy(old)
        del review["nodes"][0]["description"]
        review["nodes"][0]["nullable"] = None
        del review["nodes"][0]["properties"]["note"]
        new["nodes"][0].update(description="model", nullable="model")
        new["nodes"][0]["properties"]["confidence"] = .8
        merged = merge_review_graph(old, review, new)["nodes"][0]
        self.assertNotIn("description", merged)
        self.assertIsNone(merged["nullable"])
        self.assertEqual(merged["properties"], {"confidence": .8})

    def test_nested_graph_metadata_refreshes_without_erasing_teacher_notes(self):
        old = graph([node("a")], metadata={"quality": {"count": 1}, "note": "old"})
        review, new = deepcopy(old), deepcopy(old)
        review["metadata"]["note"] = "review"
        new["metadata"]["quality"]["count"] = 2
        merged = merge_review_graph(old, review, new)
        self.assertEqual(merged["metadata"], {"quality": {"count": 2}, "note": "review"})

    def test_existing_non_model_audit_survives_even_if_already_in_baseline(self):
        manual = {"origin": "teacher_edit", "note": "历史审核"}
        old = graph([node("a")], audit_log=[manual, {"origin": "model", "note": "旧模型日志"}])
        merged = merge_review_graph(old, deepcopy(old), graph([node("a")], audit_log=[{"origin": "model", "note": "新模型日志"}]))
        self.assertEqual(merged["audit_log"], [{"origin": "model", "note": "新模型日志"}, manual])

    def test_review_removed_candidate_stays_removed_without_deleting_its_nodes(self):
        candidate = {"source": "a", "target": "b", "relation": "PART_OF", "review_status": "pending"}
        old = graph([node("a"), node("b")], hierarchy_candidates=[candidate])
        review = deepcopy(old)
        review["hierarchy_candidates"] = []
        self.assertEqual(merge_review_graph(old, review, deepcopy(old))["hierarchy_candidates"], [])

    def test_explicit_course_conflicts_rejected_even_when_only_in_properties(self):
        old = graph([node("a")])
        review, new = deepcopy(old), deepcopy(old)
        new["nodes"][0]["properties"] = {"course_id": "another_course"}
        with self.assertRaisesRegex(ValueError, "course"):
            merge_review_graph(old, review, new)
        new = deepcopy(old)
        new["course_id"] = "another_course"
        with self.assertRaises(ValueError):
            merge_review_graph(old, review, new)

    def test_course_manifest_conflict_rejected(self):
        old = {"course": {"course_id": "a"}, "nodes": [], "links": []}
        new = {"course_manifest": {"course_id": "b"}, "nodes": [], "links": []}
        with self.assertRaises(ValueError):
            merge_review_graph(old, deepcopy(old), new)

    def test_duplicate_or_missing_node_identity_rejected(self):
        for bad in ([node("a"), node("a")], [{"name": "without id"}]):
            with self.assertRaises(ValueError):
                merge_review_graph(graph([]), graph([]), graph(bad))

    def test_inputs_are_not_mutated_and_first_model_stays_pending(self):
        old, review = graph([]), graph([])
        new = graph([node("a", properties={"review_status": "pending", "evidence": [{"id": "p"}]})])
        originals = deepcopy((old, review, new))
        merged = merge_review_graph(old, review, new)
        self.assertEqual((old, review, new), originals)
        merged["nodes"][0]["properties"]["evidence"][0]["id"] = "modified"
        self.assertEqual(new["nodes"][0]["properties"]["evidence"][0]["id"], "p")
        self.assertEqual(merged["nodes"][0]["review_status"], "pending")


if __name__ == "__main__":
    unittest.main()
