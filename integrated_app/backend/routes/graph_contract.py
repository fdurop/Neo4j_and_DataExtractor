"""Local JSON graph editing rules, independent of Flask and model/database clients."""
from __future__ import annotations

import copy
import re
import uuid
from datetime import datetime, timezone


REVIEW_STATES = {"pending", "approved", "rejected"}
TEACHING_ROLES = {"pending", "learning", "detail", "context_dependent"}


class GraphError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _payload(data, allowed):
    if not isinstance(data, dict):
        raise GraphError("请求必须是 JSON 对象")
    unknown = set(data) - set(allowed)
    if unknown:
        raise GraphError("不支持的字段：" + ", ".join(sorted(unknown)))


def _text(value, label, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise GraphError(label + "必须是" + ("字符串" if empty else "非空字符串"))
    return value.strip() if not empty else value


def _course(data):
    return _text(data["course_name"], "course_name") if "course_name" in data else None


def _state(value, label="review_status"):
    if not isinstance(value, str) or value not in REVIEW_STATES:
        raise GraphError(label + "只能是 pending、approved 或 rejected")
    return value


def _role(value):
    if not isinstance(value, str) or value not in TEACHING_ROLES:
        raise GraphError("teaching_role 不合法")
    return value


def _relation(value):
    value = _text(value, "relation")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}|[\u3400-\u9fff]{1,20}", value):
        raise GraphError("relation 需要使用大写关系代码或中文名称")
    return value


def _validate_auxiliary(data, *, role=False):
    for key in (["role_reviewed_at"] if role else ["reviewed_at"]):
        if key in data and data[key] is not None:
            text = _text(data[key], key)
            try:
                datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as exc:
                raise GraphError(key + "需要 ISO 时间字符串或 null") from exc
    if "origin" in data:
        _text(data["origin"], "origin")
    for key in ("review_required", "auto_generated"):
        if key in data and not isinstance(data[key], bool):
            raise GraphError(key + "需要布尔值")


def _properties(record):
    return record.get("properties") if isinstance(record.get("properties"), dict) else {}


def value_of(record, key, default=None):
    return record[key] if key in record else _properties(record).get(key, default)


def endpoint(value):
    if isinstance(value, dict):
        return value.get("id", value.get("name"))
    return value


def relation_of(link):
    return value_of(link, "relation") or link.get("type") or link.get("value") or "RELATED_TO"


def _sync(record, values):
    properties = record.setdefault("properties", {})
    if not isinstance(properties, dict):
        raise GraphError("记录的 properties 不是对象，请先修正图谱", 409)
    for key, value in values.items():
        record[key] = copy.deepcopy(value)
        properties[key] = copy.deepcopy(value)
        if key == "name" and "canonical_name" in properties:
            properties["canonical_name"] = value
        if key == "name" and "canonical_name" in record:
            record["canonical_name"] = value


def _graph(graphs, course):
    graph = graphs.get(course)
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("links"), list):
        raise GraphError("课程图谱不存在或格式不正确", 404)
    return graph


def _node_matches(node, selector):
    return value_of(node, "id") == selector or value_of(node, "name") == selector


def _find_node(graph, selector):
    found = [n for n in graph["nodes"] if _node_matches(n, selector)]
    if len(found) != 1:
        raise GraphError("节点不存在" if not found else "节点标识存在歧义", 404 if not found else 409)
    return found[0]


def _endpoint_owner(graph, value):
    """IDs take precedence over name fallbacks when a label looks like an ID."""
    selector = endpoint(value)
    found = [node for node in graph["nodes"] if value_of(node, "id") == selector]
    if not found:
        found = [node for node in graph["nodes"] if value_of(node, "name") == selector]
    if len(found) > 1:
        raise GraphError("关系端点存在歧义，请先修正图谱", 409)
    return found[0] if found else None


def _select_course(graphs, course, predicate):
    if course is not None:
        graph = _graph(graphs, course)
        if not predicate(graph):
            raise GraphError("指定课程中未找到目标记录", 404)
        return course, graph
    matches = [(name, graph) for name, graph in graphs.items()
               if isinstance(graph, dict) and isinstance(graph.get("nodes"), list)
               and isinstance(graph.get("links"), list) and predicate(graph)]
    if not matches:
        raise GraphError("未找到目标记录", 404)
    if len(matches) != 1:
        raise GraphError("目标存在于多个课程，请明确 course_name", 409)
    return matches[0]


def _check_unique_name(graph, name, except_node=None):
    if any(n is not except_node and value_of(n, "name") == name for n in graph["nodes"]):
        raise GraphError("该课程中已存在同名节点", 409)


def _node_values(data, *, creating=False):
    _validate_auxiliary(data, role=True)
    values = {}
    for key in ("name", "type"):
        if key in data:
            values[key] = _text(data[key], key)
    if "description" in data:
        values["description"] = _text(data["description"], "description", empty=True)
    if "teaching_role" in data:
        values["teaching_role"] = _role(data["teaching_role"])
        # A role edit without an explicit confirmation never inherits approval.
        values["role_review_status"] = "pending"
    if "role_review_status" in data:
        values["role_review_status"] = _state(data["role_review_status"], "role_review_status")
        if data["role_review_status"] == "approved" and "teaching_role" not in data:
            raise GraphError("确认教学角色时需同时传入 teaching_role")
        if data["role_review_status"] in {"approved", "rejected"}:
            values["role_reviewed_at"] = datetime.now(timezone.utc).isoformat()
    if "review_status" in data:
        values["review_status"] = _state(data["review_status"])
    if values.get("role_review_status") == "pending":
        values["role_reviewed_at"] = None
    if values:
        values["origin"] = "teacher_edit"
    if creating:
        values.setdefault("description", "")
        values.setdefault("type", "knowledge")
        values.setdefault("teaching_role", "pending")
        values.setdefault("role_review_status", "pending")
        values.setdefault("review_status", "pending")
    return values


def add_node(graphs, data):
    _payload(data, {"course_name", "name", "type", "description", "teaching_role", "role_review_status", "role_reviewed_at", "review_status", "origin"})
    course = _course(data)
    if course is None:
        raise GraphError("新增节点需要 course_name")
    values = _node_values(data, creating=True)
    if "name" not in values:
        raise GraphError("缺少 name")
    output = copy.deepcopy(graphs)
    output.setdefault(course, {"nodes": [], "links": []})
    graph = _graph(output, course)
    _check_unique_name(graph, values["name"])
    node = {}
    _sync(node, dict(values, id=str(uuid.uuid4()), origin="teacher_edit"))
    graph["nodes"].append(node)
    return output, {"success": True, "node": node, "course_name": course}


def edit_node(graphs, data):
    _payload(data, {"course_name", "id", "name", "type", "description", "teaching_role", "role_review_status", "role_reviewed_at", "review_status", "origin"})
    selector = _text(data.get("id"), "id")
    course = _course(data)
    values = _node_values(data)
    if not values:
        raise GraphError("没有要修改的字段")
    output = copy.deepcopy(graphs)
    course, graph = _select_course(output, course, lambda g: any(_node_matches(n, selector) for n in g["nodes"]))
    node = _find_node(graph, selector)
    if "name" in values:
        _check_unique_name(graph, values["name"], node)
    # Assign a stable ID to legacy name-only nodes before changing their name.
    old_name = value_of(node, "name")
    stable_id = value_of(node, "id") or old_name
    values.setdefault("id", stable_id)
    if "name" in values and values["name"] != old_name:
        previous_names = []
        for stored in (node.get("previous_names"), _properties(node).get("previous_names")):
            if isinstance(stored, list):
                previous_names.extend(name for name in stored if isinstance(name, str) and name.strip())
        if isinstance(old_name, str) and old_name.strip():
            previous_names.append(old_name)
        values["previous_names"] = list(dict.fromkeys(
            name for name in previous_names if name != values["name"]))
    legacy_endpoints = [(link, key) for link in graph["links"] for key in ("source", "target")
                        if endpoint(link.get(key)) == old_name and _endpoint_owner(graph, link.get(key)) is node]
    _sync(node, values)
    if "name" in values and stable_id != old_name:
        for link, key in legacy_endpoints:
            link[key] = stable_id
    return output, {"success": True, "node": node, "course_name": course}


def delete_node(graphs, selector, course=None):
    selector = _text(selector, "id")
    if course is not None:
        course = _text(course, "course_name")
    output = copy.deepcopy(graphs)
    course, graph = _select_course(output, course, lambda g: any(_node_matches(n, selector) for n in g["nodes"]))
    node = _find_node(graph, selector)
    remaining_links = [r for r in graph["links"] if _endpoint_owner(graph, r.get("source")) is not node
                       and _endpoint_owner(graph, r.get("target")) is not node]
    if "hierarchy_candidates" in graph:
        graph["hierarchy_candidates"] = [r for r in graph["hierarchy_candidates"]
            if _endpoint_owner(graph, r.get("source")) is not node and _endpoint_owner(graph, r.get("target")) is not node]
    graph["nodes"] = [n for n in graph["nodes"] if n is not node]
    graph["links"] = remaining_links
    return output, {"success": True, "course_name": course}


def _relation_id(link):
    return value_of(link, "relation_id") or value_of(link, "relationship_id") or link.get("id")


def _stable_endpoint(graph, value):
    owner = _endpoint_owner(graph, value)
    return (value_of(owner, "id") or value_of(owner, "name")) if owner else endpoint(value)


def _link_selector(data):
    source = _text(data["source_id"], "source_id") if "source_id" in data else None
    target = _text(data["target_id"], "target_id") if "target_id" in data else None
    rid = _text(data["relation_id"], "relation_id") if "relation_id" in data else None
    old = _text(data["old_relation"], "old_relation") if "old_relation" in data else None
    if (source is None) != (target is None) or (rid is None and source is None):
        raise GraphError("需要 relation_id 或完整的 source_id、target_id")

    def matches(link, graph):
        return ((rid is None or _relation_id(link) == rid)
                and (source is None or _stable_endpoint(graph, link.get("source")) == _stable_endpoint(graph, source))
                and (target is None or _stable_endpoint(graph, link.get("target")) == _stable_endpoint(graph, target))
                and (old is None or relation_of(link) == old))
    return matches


def _check_prerequisite_cycle(graph, proposed, ignore=None):
    if relation_of(proposed) != "PREREQUISITE_OF" or value_of(proposed, "review_status", "pending") == "rejected":
        return
    if value_of(proposed, "review_status") == "approved" and not str(value_of(proposed, "description", "")).strip():
        raise GraphError("确认先修关系需要填写教学依据")
    source, target = _stable_endpoint(graph, proposed["source"]), _stable_endpoint(graph, proposed["target"])
    adjacency = {}
    for link in graph["links"]:
        if link is ignore or relation_of(link) != "PREREQUISITE_OF" or value_of(link, "review_status", "pending") == "rejected":
            continue
        adjacency.setdefault(_stable_endpoint(graph, link.get("source")), []).append(_stable_endpoint(graph, link.get("target")))
    todo, seen = [target], set()
    while todo:
        current = todo.pop()
        if current == source:
            raise GraphError("这条先修关系会形成循环，未保存", 409)
        if current not in seen:
            seen.add(current)
            todo.extend(adjacency.get(current, []))


def add_link(graphs, data):
    _payload(data, {"course_name", "source_id", "target_id", "relation", "description", "review_status", "review_required", "reviewed_at", "origin", "auto_generated"})
    _validate_auxiliary(data)
    course = _course(data)
    source, target = _text(data.get("source_id"), "source_id"), _text(data.get("target_id"), "target_id")
    relation = _relation(data.get("relation", "RELATED_TO"))
    status = _state(data.get("review_status", "pending"))
    description = _text(data.get("description", ""), "description", empty=True)
    output = copy.deepcopy(graphs)
    course, graph = _select_course(output, course, lambda g: any(_node_matches(n, source) for n in g["nodes"]) and any(_node_matches(n, target) for n in g["nodes"]))
    source_node, target_node = _find_node(graph, source), _find_node(graph, target)
    source = value_of(source_node, "id") or value_of(source_node, "name")
    target = value_of(target_node, "id") or value_of(target_node, "name")
    if source == target:
        raise GraphError("不能连接节点自身")
    if any(_stable_endpoint(graph, r.get("source")) == source and _stable_endpoint(graph, r.get("target")) == target and relation_of(r) == relation for r in graph["links"]):
        raise GraphError("相同类型的关系已存在", 409)
    link = {"source": source, "target": target, "value": relation}
    _sync(link, {"relation_id": str(uuid.uuid4()), "relation": relation, "description": description,
                 "review_status": status, "origin": "teacher_edit",
                 "review_required": data.get("review_required", status == "pending"),
                 "auto_generated": data.get("auto_generated", False)})
    if status in {"approved", "rejected"}:
        _sync(link, {"reviewed_at": datetime.now(timezone.utc).isoformat()})
    _check_prerequisite_cycle(graph, link)
    graph["links"].append(link)
    return output, {"success": True, "link": link, "course_name": course}


def edit_link(graphs, data):
    _payload(data, {"course_name", "source_id", "target_id", "relation_id", "old_relation", "relation", "description", "review_status", "review_required", "reviewed_at", "origin", "auto_generated"})
    _validate_auxiliary(data)
    course, matches = _course(data), _link_selector(data)
    values = {}
    if "relation" in data:
        values["relation"] = _relation(data["relation"])
    if "description" in data:
        values["description"] = _text(data["description"], "description", empty=True)
    if "review_status" in data:
        values["review_status"] = _state(data["review_status"])
    if not values:
        raise GraphError("没有要修改的字段")
    output = copy.deepcopy(graphs)
    course, graph = _select_course(output, course, lambda g: any(matches(r, g) for r in g["links"]))
    found = [r for r in graph["links"] if matches(r, graph)]
    if len(found) != 1:
        raise GraphError("同端点存在多条关系，请传 relation_id 或 old_relation", 409)
    link = found[0]
    proposed = copy.deepcopy(link)
    semantic_change = any(k in values and values[k] != value_of(link, k) for k in ("relation", "description"))
    if semantic_change and "review_status" not in values:
        values["review_status"] = "pending"
    if "review_status" in values:
        values["review_required"] = values["review_status"] == "pending"
    for key in ("review_required", "auto_generated"):
        if key in data:
            values[key] = data[key]
    values["origin"] = "teacher_edit"
    if values.get("review_status") in {"approved", "rejected"}:
        values["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    elif values.get("review_status") == "pending":
        values["reviewed_at"] = None
    _sync(proposed, values)
    proposed["value"] = relation_of(proposed)
    if any(r is not link and _stable_endpoint(graph, r.get("source")) == _stable_endpoint(graph, link.get("source")) and _stable_endpoint(graph, r.get("target")) == _stable_endpoint(graph, link.get("target")) and relation_of(r) == relation_of(proposed) for r in graph["links"]):
        raise GraphError("修改后会产生相同类型的重复关系", 409)
    _check_prerequisite_cycle(graph, proposed, ignore=link)
    graph["links"][graph["links"].index(link)] = proposed
    return output, {"success": True, "link": proposed, "course_name": course}


def delete_link(graphs, data):
    _payload(data, {"course_name", "source_id", "target_id", "relation_id", "old_relation", "relation"})
    course = _course(data)
    # Legacy DELETE used relation instead of old_relation; honor it if supplied.
    selector_data = dict(data)
    if "old_relation" not in selector_data and "relation" in selector_data:
        selector_data["old_relation"] = _text(selector_data["relation"], "relation")
    matches = _link_selector(selector_data)
    output = copy.deepcopy(graphs)
    course, graph = _select_course(output, course, lambda g: any(matches(r, g) for r in g["links"]))
    found = [r for r in graph["links"] if matches(r, graph)]
    if len(found) != 1:
        raise GraphError("同端点存在多条关系，请传 relation_id 或 old_relation", 409)
    graph["links"].remove(found[0])
    return output, {"success": True, "course_name": course}
