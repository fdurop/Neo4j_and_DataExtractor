"""Pure three-way merge of a model graph and a teacher's reviewed graph.

``previous_model`` must be the unedited model snapshot that the review started
from, not a previously published review. Fields unchanged by the reviewer take
the new model value; changed fields, additions and deletions take the review.
The caller stores the new unedited model snapshot separately for the next merge.
"""
from __future__ import annotations

import copy
import json
from typing import Any


_MISSING = object()
_COLLECTIONS = {"nodes", "links", "hierarchy_candidates"}
_COUNTS = {"node_count", "link_count"}
_AUDIT_LISTS = {"review_items", "audit", "audit_log", "audit_events", "history", "edit_history", "review_history"}


def _clone(value: Any) -> Any:
    return _MISSING if value is _MISSING else copy.deepcopy(value)


def _merge_value(previous: Any, current: Any, incoming: Any) -> Any:
    """Apply the review's field-level changes over the incoming JSON value."""
    if current == previous:
        return _clone(incoming)
    if current is _MISSING:
        return _MISSING
    if isinstance(current, dict) and isinstance(incoming, dict) and (
        previous is _MISSING or isinstance(previous, dict)
    ):
        baseline = {} if previous is _MISSING else previous
        result = {}
        for key in dict.fromkeys([*incoming, *current, *baseline]):
            value = _merge_value(baseline.get(key, _MISSING), current.get(key, _MISSING), incoming.get(key, _MISSING))
            if value is not _MISSING:
                result[key] = value
        return result
    return _clone(current)


def _property(record: dict, key: str, default: Any = None) -> Any:
    if key in record:
        return record[key]
    properties = record.get("properties")
    return properties.get(key, default) if isinstance(properties, dict) else default


def _identifier(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int)):
        value = str(value).strip()
        return value or None
    return None


def _endpoint(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("id", value.get("name"))
    return _identifier(value)


def _node_key(record: dict) -> tuple:
    identity = _identifier(_property(record, "id"))
    if identity is None:
        raise ValueError("Graph nodes require a nonempty id")
    return ("node", identity)


def _edge_key(record: dict) -> tuple:
    identity = _identifier(_property(record, "id")) or _identifier(_property(record, "relation_id"))
    if identity is not None:
        return ("edge_id", identity)
    source, target = _endpoint(_property(record, "source")), _endpoint(_property(record, "target"))
    if source is None or target is None:
        raise ValueError("Graph links and candidates require source and target")
    relation = _property(record, "relation") or record.get("type") or record.get("value") or "RELATED_TO"
    if not isinstance(relation, str) or not relation.strip():
        raise ValueError("Graph relation must be a nonempty string")
    return ("edge", source, target, relation.strip())


def _records(graph: dict, field: str, key_function) -> tuple[dict, list]:
    values = graph.get(field, [])
    if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
        raise ValueError(field + " must be a list of objects")
    records, order = {}, []
    for value in values:
        key = key_function(value)
        if key in records:
            raise ValueError("Ambiguous duplicate identity in " + field)
        records[key] = value
        order.append(key)
    return records, order


def _merge_records(previous: dict, current: dict, incoming: dict, field: str, key_function) -> list[dict]:
    old, old_order = _records(previous, field, key_function)
    reviewed, review_order = _records(current, field, key_function)
    fresh, fresh_order = _records(incoming, field, key_function)
    result = []
    # Keep the review's stable order, appending newly discovered model items.
    for key in dict.fromkeys([*review_order, *fresh_order, *old_order]):
        before = old.get(key, _MISSING)
        edited = reviewed.get(key, _MISSING)
        new = fresh.get(key, _MISSING)
        if before is not _MISSING and edited is _MISSING:
            continue  # An explicit teacher deletion must not be resurrected.
        if new is _MISSING:
            if edited is not _MISSING and (before is _MISSING or edited != before):
                result.append(copy.deepcopy(edited))
            continue  # Unedited entities removed by the model stay removed.
        if edited is _MISSING:
            result.append(copy.deepcopy(new))
        else:
            merged = _merge_value(before, edited, new)
            if merged is not _MISSING:
                result.append(merged)
    return result


def _course_ids(graph: dict) -> set[str]:
    identities = set()

    def include(record):
        if isinstance(record, dict):
            identity = _identifier(_property(record, "course_id"))
            if identity is not None:
                identities.add(identity)

    include(graph)
    for name in ("course", "course_manifest", "metadata"):
        value = graph.get(name)
        include(value)
        if name == "course" and isinstance(value, dict) and not _property(value, "course_id"):
            identity = _identifier(value.get("id"))
            if identity is not None:
                identities.add(identity)
    for field in ("nodes", "links", "hierarchy_candidates", "documents"):
        values = graph.get(field, [])
        if isinstance(values, list):
            for value in values:
                include(value)
    return identities


def _merge_audit(previous: list, current: list, incoming: list) -> list:
    """Refresh model audit items while retaining review additions and edits."""
    def signature(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def non_model_record(value):
        if not isinstance(value, dict):
            return False
        origin = _property(value, "origin")
        if isinstance(origin, str) and origin.strip():
            origin = origin.lower().strip()
            if not origin.startswith(("model", "auto", "llm", "machine")) and origin != "derived":
                return True
        if _property(value, "auto_generated") is False or value.get("reviewer") or value.get("actor"):
            return True
        return any(non_model_record(item) for item in value.values() if isinstance(item, dict))

    old = {signature(item) for item in previous}
    additions = [item for item in current if signature(item) not in old or non_model_record(item)]
    result, seen = [], set()
    for item in [*incoming, *additions]:
        key = signature(item)
        if key not in seen:
            result.append(copy.deepcopy(item))
            seen.add(key)
    return result


def _valid_endpoint_names(nodes: list[dict]) -> set[str]:
    valid = {_node_key(node)[1] for node in nodes}
    # Older editors also used unambiguous names as relation endpoints.
    names = {}
    for node in nodes:
        name = _identifier(_property(node, "name"))
        if name is not None:
            names[name] = names.get(name, 0) + 1
    valid.update(name for name, count in names.items() if count == 1)
    return valid


def merge_review_graph(previous_model: dict, current_review: dict, new_model: dict) -> dict:
    """Merge one course's graphs without mutating arguments or approving models.

    All explicit course IDs, including IDs on node properties, must agree.
    Lists other than graph records and audit logs are JSON field values: a
    reviewer-modified evidence list is retained, while an untouched one updates.
    Review records referring to deleted nodes remain as audit evidence; actual
    links and hierarchy candidates with missing endpoints are removed.
    """
    graphs = (previous_model, current_review, new_model)
    if any(not isinstance(graph, dict) for graph in graphs):
        raise ValueError("Three graph objects are required")
    # Validate the JSON contract, including finite numbers, before merging.
    for graph in graphs:
        json.dumps(graph, ensure_ascii=False, allow_nan=False)
    identities = set().union(*(_course_ids(graph) for graph in graphs))
    if len(identities) > 1:
        raise ValueError("Cannot merge graphs with different explicit course IDs")
    metadata = [
        {key: value for key, value in graph.items() if key not in _COLLECTIONS | _COUNTS}
        for graph in graphs
    ]
    result = _merge_value(*metadata)
    for field in _AUDIT_LISTS:
        values = [graph.get(field, []) for graph in graphs]
        if any(field in graph for graph in graphs) and all(isinstance(value, list) for value in values):
            result[field] = _merge_audit(*values)
    result["nodes"] = _merge_records(*graphs, "nodes", _node_key)
    valid = _valid_endpoint_names(result["nodes"])

    def existing_endpoints(record):
        return _endpoint(_property(record, "source")) in valid and _endpoint(_property(record, "target")) in valid

    result["links"] = [record for record in _merge_records(*graphs, "links", _edge_key) if existing_endpoints(record)]
    if any("hierarchy_candidates" in graph for graph in graphs):
        result["hierarchy_candidates"] = [record for record in _merge_records(*graphs, "hierarchy_candidates", _edge_key)
                                           if existing_endpoints(record)]
    result["node_count"], result["link_count"] = len(result["nodes"]), len(result["links"])
    return result
