"""Offline quality checks and teacher-view export; no API/database required.

Importance scores, document membership, concept taxonomy and prerequisites are
different relations. In particular, evidence overlap is only a review candidate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from relation_review import partition_relationships

HIERARCHY_TYPES = {"HAS_PART", "PART_OF", "IS_A"}


def _restore_review_properties(properties, *, concept=False):
    """Restore Browser table metadata without importing the database client."""
    restored = copy.deepcopy(properties)
    for field in (["review_status", "role_review_status"] if concept else ["review_status"]):
        if restored.get(field) not in ("approved", "rejected", "pending"):
            restored[field] = "pending"
    if concept:
        raw = restored.get("page_observations_json", restored.get("page_observations", []))
        try:
            observations = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(observations, list) or not all(isinstance(o, dict) for o in observations):
                raise ValueError("expected a list of objects")
            json.dumps(observations, allow_nan=False)
        except (ValueError, TypeError, RecursionError):
            observations = []
            restored["metadata_restore_warnings"] = ["invalid_page_observations_json"]
        restored["page_observations"] = observations
    return restored


def import_neo4j_table(rows):
    """Deduplicate Browser table rows without interpreting A/C as edge direction.

    Neo4j identity values can exceed JS's safe integer range. Use elementId (or
    Python integer strings) to resolve physical endpoints, not array positions.
    """
    nodes, relationships = {}, {}

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            if isinstance(value.get("labels"), list) and isinstance(value.get("properties"), dict):
                nodes[str(value.get("elementId", value.get("identity")))] = value
            elif "type" in value and "start" in value and "end" in value:
                relationships[str(value.get("elementId", value.get("identity")))] = value
            else:
                for item in value.values():
                    walk(item)

    walk(rows)
    by_identity = {str(node.get("identity")): node for node in nodes.values()}
    concepts = []
    formal = {"documents": [], "pages": [], "multimodal_nodes": [], "course_manifest": {}, "relationships": []}
    for node in nodes.values():
        props = copy.deepcopy(node["properties"])
        labels = node["labels"]
        if "Concept" in labels:
            props = _restore_review_properties(props, concept=True)
            props["canonical_name"] = _name(props.get("canonical_name") or props.get("name") or props.get("id"))
            concepts.append(props)
        elif "Document" in labels:
            props.setdefault("document_id", props.get("id"))
            formal["documents"].append(props)
        elif "Page" in labels:
            props.setdefault("page_id", props.get("id"))
            formal["pages"].append(props)
        elif "Course" in labels:
            formal["course_manifest"] = props
        else:
            props.setdefault("node_id", props.get("id"))
            formal["multimodal_nodes"].append(props)
    semantic, unresolved = [], []
    for identity, edge in relationships.items():
        source = nodes.get(str(edge.get("startNodeElementId"))) or by_identity.get(str(edge["start"]))
        target = nodes.get(str(edge.get("endNodeElementId"))) or by_identity.get(str(edge["end"]))
        if source is None or target is None:
            unresolved.append(identity)
            continue
        if "Concept" not in source["labels"] or "Concept" not in target["labels"]:
            if "Concept" not in source["labels"] and "Concept" not in target["labels"]:
                props = copy.deepcopy(edge.get("properties", {}))
                formal["relationships"].append({
                    "relationship_id": props.get("relationship_id") or identity,
                    "source_id": source["properties"].get("id"),
                    "target_id": target["properties"].get("id"),
                    "relationship_type": edge["type"], "properties": props,
                })
            continue
        props = _restore_review_properties(edge.get("properties", {}))
        semantic.append(dict(props, source=_name(source["properties"].get("canonical_name") or source["properties"].get("name")),
                             target=_name(target["properties"].get("canonical_name") or target["properties"].get("name")),
                             relation=edge["type"], neo4j_element_id=identity))
    graph = {"concepts": concepts, "relationships": semantic, "import_metadata": {
        "format": "neo4j_browser_table", "row_count": len(rows),
        "unique_node_count": len(nodes), "unique_relationship_count": len(relationships),
        "node_labels": dict(Counter(label for node in nodes.values() for label in node["labels"])),
        "all_relation_types": dict(Counter(r["type"] for r in relationships.values())),
        "unresolved_relationships": unresolved,
        "scope_note": "Counts describe this export, not necessarily the full database. Check the original query LIMIT.",
    }}
    return graph, formal


def _name(value):
    return str(value or "").strip()


def _unique(values):
    if isinstance(values, str):
        values = [values]
    return list(dict.fromkeys(_name(v) for v in (values or []) if _name(v)))


def _cycle_nodes(edges):
    """Return exact cycle participants, rather than downstream blocked nodes."""
    adjacency = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
    result = set()
    # Each edge u->v is cyclic iff v can reach u. Avoid recursion on large graphs.
    for a, b in edges:
        todo, seen = [b], set()
        while todo:
            current = todo.pop()
            if current == a:
                result.update((a, b))
                break
            if current not in seen:
                seen.add(current)
                todo.extend(adjacency[current] - seen)
    return sorted(result)


def hierarchy_pair(relation):
    kind = relation.get("relation")
    if kind == "HAS_PART":
        return relation["source"], relation["target"]
    if kind in {"PART_OF", "IS_A"}:
        return relation["target"], relation["source"]
    return None


def prepare_knowledge(data):
    """Copy and validate a graph, retaining rejected records for manual review.

    Accepts native concepts_graph_v2 or a frontend nodes/links export. Unverified
    auto-parent edges are removed from factual relationships, including caches.
    """
    if isinstance(data, list):
        data, _ = import_neo4j_table(data)
    output = copy.deepcopy(data)
    if "concepts" not in output:
        raw_nodes = output.get("nodes", [])
        concepts = []
        ids = {}
        for node in raw_nodes:
            props = node.get("properties") or {}
            labels = node.get("labels") or props.get("labels") or []
            kind = node.get("type") or props.get("node_type")
            if "Concept" not in labels and kind not in {"concept", "Concept", "knowledge"}:
                continue
            item = dict(props)
            item.update(node)
            item["canonical_name"] = _name(node.get("name") or props.get("name"))
            ids[_name(node.get("id"))] = item["canonical_name"]
            concepts.append(item)
        output["concepts"] = concepts
        output["relationships"] = [
            dict(edge, source=ids[_name(edge.get("source"))],
                 target=ids[_name(edge.get("target"))],
                 relation=edge.get("relation") or edge.get("type") or "RELATED_TO")
            for edge in output.get("links", [])
            if _name(edge.get("source")) in ids and _name(edge.get("target")) in ids
        ]
    candidates = output.get("hierarchy_candidates", [])
    rejected = output.get("review_items", [])
    valid_concepts, seen = [], set()
    for item in output.get("concepts", []):
        name = _name(item.get("canonical_name"))
        if not name or name in seen:
            rejected.append({"reason": "empty_or_duplicate_concept", "record": item})
            continue
        seen.add(name)
        item = _restore_review_properties(item, concept=True)
        item["canonical_name"] = name
        item["importance_level"] = item.get("importance_level", item.get("layer", 3))
        item["id"] = item.get("id") or "concept_" + hashlib.sha256(name.encode()).hexdigest()[:16]
        valid_concepts.append(item)
    valid_relations, signatures = [], set()
    for relation in output.get("relationships", []):
        relation = _restore_review_properties(relation)
        relation["source"] = _name(relation.get("source"))
        relation["target"] = _name(relation.get("target"))
        kind = _name(relation.get("relation"))
        if (relation.get("auto_generated") and kind in HIERARCHY_TYPES | {"HAS_PARAMETER"}
                and relation.get("review_status") not in {"approved", "rejected"}):
            candidate = dict(relation, review_status="pending", review_required=True,
                             candidate_reason="evidence_overlap_does_not_prove_semantic_hierarchy")
            if candidate not in candidates:
                candidates.append(candidate)
            continue
        signature = (relation["source"], relation["target"], kind)
        if relation["source"] not in seen or relation["target"] not in seen:
            rejected.append({"reason": "missing_endpoint", "record": relation})
        elif relation["source"] == relation["target"]:
            rejected.append({"reason": "self_relation", "record": relation})
        elif signature not in signatures:
            signatures.add(signature)
            valid_relations.append(relation)
    valid_relations, semantic_review_items = partition_relationships(valid_relations)
    rejected.extend(item for item in semantic_review_items if item not in rejected)
    output.update(concepts=valid_concepts, relationships=valid_relations,
                  hierarchy_candidates=candidates, review_items=rejected,
                  layer_semantics="importance_only_not_semantic_depth")
    hierarchy = [hierarchy_pair(r) for r in valid_relations
                 if r.get("relation") in HIERARCHY_TYPES and r.get("review_status") != "rejected"]
    hierarchy_set = set(hierarchy)
    reverse_pairs = sorted({tuple(sorted((a, b))) for a, b in hierarchy_set if (b, a) in hierarchy_set})
    prerequisites = [(r["source"], r["target"]) for r in valid_relations
                     if r.get("relation") == "PREREQUISITE_OF" and r.get("review_status") != "rejected"]
    related = {n for r in valid_relations for n in (r["source"], r["target"])}
    name_owners = {c["canonical_name"].casefold(): c["canonical_name"] for c in valid_concepts}
    alias_name_conflicts, alias_owners = [], defaultdict(set)
    for concept in valid_concepts:
        for alias in _unique(concept.get("aliases")):
            alias_owners[alias.casefold()].add(concept["canonical_name"])
            other = name_owners.get(alias.casefold())
            if other and other != concept["canonical_name"]:
                alias_name_conflicts.append({"concept": concept["canonical_name"], "alias": alias, "other_concept": other})
    report = {
        "concept_count": len(valid_concepts), "relationship_count": len(valid_relations),
        "relation_types": dict(Counter(r.get("relation") for r in valid_relations)),
        "importance_distribution": dict(Counter(str(c["importance_level"]) for c in valid_concepts)),
        "suggested_role_distribution": dict(Counter(c.get("suggested_teaching_role", "pending") for c in valid_concepts)),
        "confirmed_learning_count": sum(c.get("role_review_status") == "approved"
                                        and c.get("teaching_role") == "learning" for c in valid_concepts),
        "pending_hierarchy_count": len(candidates), "review_item_count": len(rejected),
        "hierarchy_cycle_nodes": _cycle_nodes(hierarchy),
        "opposing_hierarchy_pairs": reverse_pairs,
        "prerequisite_cycle_nodes": _cycle_nodes(prerequisites),
        "alias_name_conflicts": alias_name_conflicts,
        "shared_aliases": {a: sorted(owners) for a, owners in alias_owners.items() if len(owners) > 1},
        "large_alias_groups_for_review": [{"concept": c["canonical_name"], "aliases": c.get("aliases", [])}
                                           for c in valid_concepts if len(c.get("aliases", [])) >= 10],
        "isolated_concepts": sorted(seen - related),
        "concepts_without_evidence": [c["canonical_name"] for c in valid_concepts
                                      if not c.get("evidence_node_ids") and not c.get("source_page_ids")],
    }
    if output.get("import_metadata"):
        report["import_metadata"] = output["import_metadata"]
    return output, report


def prerequisite_path(data, target, mastered=(), approved_only=True):
    """Return a study order for an acyclic, reviewed prerequisite subgraph.

    A runtime REQUIRES relation is deliberately not treated as a learning edge.
    Mastered concepts are omitted but their ancestors are still traversed.
    """
    prepared, _ = prepare_knowledge(data)
    names = {c["canonical_name"] for c in prepared["concepts"]}
    if target not in names:
        raise ValueError("Unknown target concept")
    parents = defaultdict(set)
    for relation in prepared["relationships"]:
        if relation.get("relation") != "PREREQUISITE_OF":
            continue
        if approved_only and relation.get("review_status") != "approved":
            continue
        parents[relation["target"]].add(relation["source"])
    needed, todo = set(), [target]
    while todo:
        node = todo.pop()
        if node not in needed:
            needed.add(node)
            todo.extend(parents[node])
    edges = [(p, n) for n in needed for p in parents[n] if p in needed]
    cycles = _cycle_nodes(edges)
    if cycles:
        return {"status": "cycle_requires_review", "cycle_nodes": cycles, "order": []}
    children, indegree = defaultdict(set), {n: 0 for n in needed}
    for source, destination in edges:
        children[source].add(destination)
        indegree[destination] += 1
    queue = deque(sorted(n for n, degree in indegree.items() if not degree))
    order = []
    while queue:
        node = queue.popleft()
        if node not in set(mastered) and node != target:
            order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if not indegree[child]:
                queue.append(child)
    return {"status": "ok", "target": target, "order": order,
            "direct_prerequisites": sorted(parents[target])}


def export_teacher_bundle(data, formal_data, output_dir):
    """Export a portable editable review view; preserve the source graph."""
    graph, report = prepare_knowledge(data)
    documents = formal_data.get("documents", [])
    pages = {p["page_id"]: p for p in formal_data.get("pages", [])}
    evidence_nodes = {n.get("node_id"): n
                      for n in formal_data.get("multimodal_nodes", formal_data.get("nodes", []))}
    evidence_pages = {key: n.get("source_page_id") for key, n in evidence_nodes.items()}
    known_documents = {d["document_id"] for d in documents}
    def material_reference(evidence_id, node):
        reference = {"evidence_id": evidence_id, "node_type": node.get("node_type", "Material"),
                     "text": node.get("description") or node.get("raw_text") or "", "review_status": "pending"}
        # Preserve the distinction between page-level observations and local
        # figures, including claim warnings; older exports need no new defaults.
        for key in ("source_kind", "description_scope", "semantic_review_flags", "task_id",
                    "fresh_run", "source_image_sha256", "same_run_occurrence_reuse"):
            if key in node:
                reference[key] = copy.deepcopy(node[key])
        return reference

    def sources_for(record):
        page_ids = _unique(list(record.get("source_page_ids") or []) + [
            evidence_pages.get(e) or (e if e in pages else None) for e in record.get("evidence_node_ids", [])
        ])
        return [
            {"page_id": p, "document_id": pages[p].get("document_id"),
             "page_no": pages[p].get("page_no"), "title": pages[p].get("title", ""),
             # Do not substitute a model summary when the original text is absent.
             "extracted_text": pages[p].get("raw_text") or "",
             "material_evidence": [
                 material_reference(evidence_id, node)
                 for evidence_id in _unique(record.get("evidence_node_ids"))
                 if (node := evidence_nodes.get(evidence_id))
                 and node.get("source_page_id") == p
                 and node.get("node_type") not in {"Page", "TextChunk", "Text"}
             ]}
            for p in page_ids if p in pages
        ]

    for concept in graph["concepts"]:
        concept["sources"] = sources_for(concept)
        counts = Counter(s["document_id"] for s in concept["sources"] if s["document_id"] in known_documents)
        if not concept.get("chapter_id"):
            concept["chapter_id"] = counts.most_common(1)[0][0] if counts else ""
    for relationship in graph["relationships"] + graph["hierarchy_candidates"]:
        relationship["sources"] = sources_for(relationship)
    for item in graph["review_items"]:
        record = item.get("record")
        if isinstance(record, dict):
            record["sources"] = sources_for(record)
    graph.update(documents=documents, course=formal_data.get("course_manifest", formal_data.get("manifest", {})), quality=report,
                 view_note="Document groups are source organization; importance is not taxonomy depth.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "teacher_graph.json"
    json_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "hierarchy_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    template = Path(__file__).with_name("teacher_workbench.html").read_text(encoding="utf-8")
    embedded = json.dumps(graph, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html_path = output_dir / "teacher_workbench.html"
    html_path.write_text(template.replace("/*__INITIAL_GRAPH__*/null", embedded), encoding="utf-8")
    return {"graph": str(json_path), "audit": str(output_dir / "hierarchy_audit.json"), "workbench": str(html_path)}


def main():
    parser = argparse.ArgumentParser(description="离线审核图谱并生成教师编辑页面，不调用模型或数据库")
    parser.add_argument("--graph", required=True, help="concepts_graph_v2.json 或包含 concepts/relationships 的 JSON")
    parser.add_argument("--kg-data", type=Path, help="原生概念 JSON 的证据目录；Neo4j 表格导出可不填")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    graph = json.loads(Path(args.graph).read_text(encoding="utf-8-sig"))
    formal = {}
    if isinstance(graph, list):
        graph, formal = import_neo4j_table(graph)
    if args.kg_data:
        for name in ("documents", "pages", "multimodal_nodes", "course_manifest"):
            path = args.kg_data / f"{name}.json"
            formal[name] = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else ({} if name == "course_manifest" else [])
    elif not formal:
        parser.error("原生概念 JSON 需要 --kg-data；Neo4j 表格导出可自行恢复证据")
    print(json.dumps(export_teacher_bundle(graph, formal, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
