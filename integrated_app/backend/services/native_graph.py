"""Course-scoped native graph reads and transactional review deltas."""
import copy
import base64
import hashlib
import json
import re
import zlib
import time
from collections import defaultdict
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from backend.routes.graph_contract import GraphError, relation_of, endpoint

SNAPSHOTS = {'review_graph_json', 'model_graph_json', 'previous_review_graph_json',
             'review_revision', 'review_updated_at', 'native_graph_version', 'undo_revision'}
LABELS = {'concept': 'Concept', 'knowledge': 'Concept', 'course': 'Course',
          'document': 'Document', 'page': 'Page', 'figure': 'Figure', 'table': 'Table',
          'formula': 'Formula', 'code_block': 'CodeBlock', 'text_chunk': 'TextChunk'}
FIELDS = {'id', 'name', 'canonical_name', 'description', 'layer', 'category', 'importance_level',
          'review_status', 'origin', 'review_required', 'auto_generated', 'reviewed_at',
          'teaching_role', 'role_review_status', 'role_reviewed_at', 'previous_names',
          'relation_id', 'relation'}


def encode_snapshot(graph):
    data = json.dumps(graph, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    return json.dumps({'_graph_encoding': 'zlib-base64-v1',
                      'data': base64.b64encode(zlib.compress(data)).decode('ascii')}, separators=(',', ':'))


def decode_snapshot(value):
    graph = json.loads(value) if value else {'nodes': [], 'links': []}
    if graph.get('_graph_encoding') == 'zlib-base64-v1':
        graph = json.loads(zlib.decompress(base64.b64decode(graph['data'])))
    return graph


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,63}|[\u3400-\u9fff]{1,20}', value):
        raise GraphError('数据库标签或关系类型无效', 409)
    return '`' + value + '`'


def props(record):
    result = copy.deepcopy(record.get('properties') or {})
    result.update({k: v for k, v in record.items() if k in FIELDS})
    result = {k: v for k, v in result.items() if k not in SNAPSHOTS and not k.startswith('_app_')}
    for key, value in list(result.items()):
        if isinstance(value, dict) or (isinstance(value, list) and any(isinstance(x, (dict, list)) for x in value)):
            result.pop(key)
            result[key + '_json'] = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return result


def nodes(graph):
    result = {}
    for node in graph.get('nodes', []):
        identity = node.get('id') or (node.get('properties') or {}).get('id')
        if not identity or identity in result:
            raise GraphError('节点标识缺失或重复，未写入数据库', 409)
        result[identity] = node
    return result


def links(graph):
    result = {}
    for link in graph.get('links', []):
        key = (endpoint(link['source']), endpoint(link['target']), relation_of(link))
        if key in result:
            raise GraphError('同一方向、类型存在重复关系，请先清理重复关系', 409)
        result[key] = link
    return result


def _run(tx, query, **params):
    return tx.run(query, **params)


def validate_course_graph(graph, course_id):
    """Reject a foreign or malformed snapshot before any write, including undo."""
    if not isinstance(course_id, str) or not course_id.strip():
        raise GraphError('修改必须明确指定课程', 400)
    members = nodes(graph)
    for node in members.values():
        for record in (node, node.get('properties') or {}):
            if record.get('course_id') not in (None, course_id):
                raise GraphError('图谱含其他课程的节点，未保存', 409)
        if node.get('type') == 'course' or 'Course' in node.get('labels', []):
            if node.get('id') != course_id:
                raise GraphError('课程根节点标识不匹配，未保存', 409)
    for (source, target, kind), link in links(graph).items():
        token(kind)
        if source not in members or target not in members:
            raise GraphError('关系端点不属于当前课程图谱，未保存', 409)
        for record in (link, link.get('properties') or {}):
            if record.get('course_id') not in (None, course_id):
                raise GraphError('图谱含其他课程的关系，未保存', 409)


def apply_delta(tx, before, after, course_id, force=False):
    """All changes share the caller's revision transaction. Never touch other courses."""
    validate_course_graph(before, course_id)
    validate_course_graph(after, course_id)
    old_nodes, new_nodes = nodes(before), nodes(after)
    old_links, new_links = links(before), links(after)
    removed_nodes = list(set(old_nodes) - set(new_nodes))
    if course_id in removed_nodes:
        raise GraphError('不能删除课程根节点', 409)
    if removed_nodes:
        # DETACH would also erase an anomalous edge owned by a different course.
        outside = _run(tx, 'MATCH (n {course_id:$course})-[r]-(peer) '
            'WHERE n.id IN $ids AND coalesce(peer.course_id, "") <> $course '
            'RETURN count(r) AS outside', course=course_id, ids=removed_nodes).single()
        if outside and outside.get('outside', 0):
            raise GraphError('节点存在跨课程或未归属课程的关系，未删除，请先检查数据', 409)
    removed_links = set(old_links) - set(new_links)
    for source, target, kind in removed_links:
        query = (f'MATCH (a {{course_id:$course,id:$source}})-[r:{token(kind)}]->'
                 '(b {course_id:$course,id:$target}) DELETE r')
        _run(tx, query, course=course_id, source=source, target=target).consume()
    if removed_nodes:
        _run(tx, 'MATCH (n {course_id:$course}) WHERE n.id IN $ids AND NOT n:Course DETACH DELETE n',
             course=course_id, ids=removed_nodes).consume()
    groups = defaultdict(list)
    for identity, node in new_nodes.items():
        if not force and identity in old_nodes and props(node) == props(old_nodes[identity]):
            continue
        values = props(node)
        if values.get('course_id') not in (None, course_id):
            raise GraphError('拒绝跨课程写入节点', 409)
        values.update(id=identity, course_id=course_id)
        if values.get('name') and ('canonical_name' in values or node.get('type') in {'concept', 'knowledge'}):
            values['canonical_name'] = values['name']
        label = LABELS.get(node.get('type'), 'Concept')
        values['_app_type'] = node.get('type') or 'concept'
        previous = props(old_nodes[identity]) if identity in old_nodes else {}
        values.update({k: None for k in previous.keys() - values.keys() if k not in {'course_id', 'id'}})
        groups[label].append({'id': identity, 'props': values})
    for label, rows in groups.items():
        row = _run(tx, f'UNWIND $rows AS row MERGE (n:{token(label)} {{course_id:$course,id:row.id}}) '
                   'SET n += row.props RETURN count(n) AS count', rows=rows, course=course_id).single()
        if row['count'] != len(rows):
            raise GraphError('数据库节点标识不唯一，事务已取消', 409)
    groups = defaultdict(list)
    for key, link in new_links.items():
        if not force and key in old_links and props(link) == props(old_links[key]):
            continue
        source, target, kind = key
        if source not in new_nodes or target not in new_nodes:
            raise GraphError('关系端点不存在，事务已取消', 409)
        values = props(link)
        previous = props(old_links[key]) if key in old_links else {}
        values.update({k: None for k in previous.keys() - values.keys()})
        values['relation'] = kind
        values.setdefault('relation_id', hashlib.sha256((course_id + ':' + repr(key)).encode()).hexdigest())
        groups[kind].append({'source': source, 'target': target, 'props': values})
    for kind, rows in groups.items():
        row = _run(tx, 'UNWIND $rows AS row MATCH (a {course_id:$course,id:row.source}), '
                   '(b {course_id:$course,id:row.target}) '
                   f'MERGE (a)-[r:{token(kind)}]->(b) SET r += row.props RETURN count(r) AS count',
                   course=course_id, rows=rows).single()
        if row['count'] != len(rows):
            raise GraphError('数据库关系或端点不唯一，事务已取消', 409)


def _read_pages(session, query, course_id, limit=200):
    after = ''
    while True:
        for attempt in range(3):
            try:
                page = list(session.run(query, course=course_id, after=after, limit=limit))
                break
            except (ServiceUnavailable, SessionExpired):
                if attempt == 2: raise
                time.sleep(attempt + 1)
        yield from page
        if len(page) < limit: break
        cursor = page[-1]['cursor']
        if cursor <= after: raise GraphError('数据库分页没有前进', 503)
        after = cursor


def read_graph(session, template, course_id):
    """Snapshot is only an envelope; existence and editable fields come from native entities."""
    graph = copy.deepcopy(template)
    old_nodes, old_links = nodes(template), links(template)
    output_nodes = []
    rows = _read_pages(session, 'MATCH (n {course_id:$course}) WHERE n.id IS NOT NULL AND elementId(n)>$after '
                       'RETURN elementId(n) AS cursor,labels(n) AS labels, n {.*, review_graph_json:null, model_graph_json:null, '
                       'previous_review_graph_json:null, review_revision:null, review_updated_at:null} AS props '
                       'ORDER BY cursor LIMIT $limit', course_id)
    for row in rows:
        values = {k: v for k, v in dict(row['props']).items() if k not in SNAPSHOTS}
        identity = values['id']
        node = copy.deepcopy(old_nodes.get(identity, {'id': identity}))
        kind = values.pop('_app_type', None) or node.get('type')
        if not kind:
            kind = next((k for k, v in LABELS.items() if v in row['labels']), 'concept')
        for key in list(values):
            if key.endswith('_json') and key[:-5] in {'page_observations', 'previous_names'}:
                try: values[key[:-5]] = json.loads(values[key])
                except (TypeError, ValueError): pass
        node.update(type=kind, labels=list(row['labels']), properties=values)
        # Native absence clears deleted fields rather than reviving snapshot text.
        for key in FIELDS - {'relation', 'relation_id'}:
            if key in values: node[key] = copy.deepcopy(values[key])
            elif key in node and key not in {'id', 'name'}: node.pop(key)
        node['name'] = values.get('name') or values.get('canonical_name') or node.get('name') or identity
        node.setdefault('description', '')
        output_nodes.append(node)
    graph['nodes'] = output_nodes
    output_links = []
    rows = _read_pages(session, 'MATCH (a {course_id:$course})-[r]->(b {course_id:$course}) '
                       'WHERE a.id IS NOT NULL AND b.id IS NOT NULL AND elementId(r)>$after '
                       'RETURN elementId(r) AS cursor,a.id AS source,b.id AS target,'
                       'type(r) AS kind,properties(r) AS props ORDER BY cursor LIMIT $limit', course_id, 400)
    seen = set()
    for row in rows:
        key = row['source'], row['target'], row['kind']
        if key in seen: raise GraphError('数据库存在同端点同类型重复关系，请先处理', 409)
        seen.add(key)
        link = copy.deepcopy(old_links.get(key, {}))
        values = dict(row['props'])
        for field in FIELDS:
            if field in values: link[field] = values[field]
            else: link.pop(field, None)
        link.update(source=key[0], target=key[1], relation=key[2], value=key[2], properties=values)
        output_links.append(link)
    graph['links'] = output_links
    valid = {n['id'] for n in output_nodes}
    graph['hierarchy_candidates'] = [x for x in graph.get('hierarchy_candidates', [])
        if endpoint(x.get('source')) in valid and endpoint(x.get('target')) in valid]
    graph.update(node_count=len(output_nodes), link_count=len(output_links))
    return graph
