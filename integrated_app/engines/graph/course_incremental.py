"""Pure provenance patches for additive imports into one course.

These patches contain evidence metadata only. They must not carry a model's
description, teaching role or approval status through a teacher-content lock.
"""
import copy
import json


SOURCE_LIST_FIELDS = ("evidence_node_ids", "source_page_ids", "source_document_ids")


def _strings(value, label):
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(label + " must be a list of strings")
    return list(dict.fromkeys(item for item in value if item))


def _records(value, label):
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        value = []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(label + " must be a list of objects")
    json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def _signature(record):
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _union_records(previous, incoming):
    result, seen = [], set()
    for record in [*previous, *incoming]:
        signature = _signature(record)
        if signature not in seen:
            seen.add(signature)
            result.append(copy.deepcopy(record))
    return result


def concept_provenance(previous, incoming):
    """Accumulate sources without modifying either input or teacher content."""
    result = {}
    for field in SOURCE_LIST_FIELDS:
        result[field] = list(dict.fromkeys(
            _strings(previous.get(field), field) + _strings(incoming.get(field), field)
        ))
    result["evidence_count"] = len(result["evidence_node_ids"])
    observations = _union_records(
        _records(previous.get("page_observations_json", previous.get("page_observations")), "page_observations"),
        _records(incoming.get("page_observations_json", incoming.get("page_observations")), "page_observations"),
    )
    result["page_observations_json"] = json.dumps(observations, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return result


def course_audit_records(previous, incoming, *, candidates=False):
    """Retain earlier documents' audit records; coalesce duplicate candidates.

    Candidate identities follow their displayed source, target and relation,
    avoiding duplicate identities in the later three-way snapshot merge.
    No fuzzy term matching or automatic relationship approval happens here.
    """
    previous = _records(previous, "course audit records")
    incoming = _records(incoming, "course audit records")
    if not candidates:
        return _union_records(previous, incoming)
    result, indices = [], {}
    for record in [*previous, *incoming]:
        source, target = record.get("source"), record.get("target")
        relation = record.get("relation") or record.get("type") or record.get("value") or "RELATED_TO"
        key = (source, target, relation) if all(isinstance(x, str) and x for x in (source, target, relation)) else ("raw", _signature(record))
        if key not in indices:
            indices[key] = len(result)
            result.append(copy.deepcopy(record))
            continue
        retained = result[indices[key]]
        # Existing candidate text/status is retained. Importing another PDF
        # adds provenance, rather than changing an earlier review decision.
        for field, value in record.items():
            if field not in retained:
                retained[field] = copy.deepcopy(value)
        for field in SOURCE_LIST_FIELDS:
            if field in retained or field in record:
                retained[field] = list(dict.fromkeys(
                    _strings(retained.get(field), field) + _strings(record.get(field), field)
                ))
        if "evidence_node_ids" in retained:
            retained["evidence_count"] = len(retained["evidence_node_ids"])
    return result
