"""Read-only Neo4j deletion check. Distinguishes reviewed snapshots and native nodes.

Run with the project's .venv Python; requires neo4j and certifi.
Credentials are read from --env, never passed as command-line arguments.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

DEFAULT_PROJECT = Path(r'C:\Users\Lin\Desktop\study_documents\fdurop\python_doc')


def read_env(path):
    values = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.lstrip().startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip().removesuffix('_2')] = value.strip().strip('\"').strip("'")
    return values


def node_value(node, key):
    return node.get(key) or (node.get('properties') or {}).get(key)


def inspect_snapshot(graph, node_id=None, name=None):
    nodes = graph.get('nodes', [])
    matches = [n for n in nodes if (str(node_value(n, 'id')) == node_id if node_id else
                                   (node_value(n, 'name') or node_value(n, 'canonical_name')) == name)]
    identities = {str(node_value(n, 'id')) for n in matches}
    links = graph.get('links', [])
    def endpoint(value):
        return str(value.get('id', value.get('name'))) if isinstance(value, dict) else str(value)
    touching = sum(endpoint(e.get('source')) in identities or endpoint(e.get('target')) in identities for e in links)
    dangling = sum(endpoint(e.get('source')) == node_id or endpoint(e.get('target')) == node_id for e in links) if node_id else None
    return {'matching_nodes': len(matches), 'matching_node_ids': sorted(identities),
            'incident_relationships': touching, 'references_to_requested_id': dangling,
            'total_nodes': len(nodes), 'total_relationships': len(links)}


def query_database(driver, database, course_id, node_id=None, name=None):
    # One transaction provides a consistent read of both representations.
    with driver.session(database=database, default_access_mode='READ') as session:
        with session.begin_transaction(timeout=25) as tx:
            rows = list(tx.run('MATCH (c:Course) WHERE c.id=$course OR c.course_id=$course '
                               'RETURN c.id AS id, c.course_name AS name, '
                               'c.review_revision AS revision, c.review_graph_json AS graph', course=course_id))
            if len(rows) != 1 or not rows[0]['graph']:
                raise ValueError('Course or reviewed snapshot missing/ambiguous; absence cannot prove deletion')
            row = rows[0]
            graph = json.loads(row['graph'])
            reviewed = inspect_snapshot(graph, node_id, name)
            native = list(tx.run('MATCH (n) WHERE n.course_id=$course AND '
                                 '(($id IS NOT NULL AND n.id=$id) OR '
                                 '($id IS NULL AND (n.name=$name OR n.canonical_name=$name))) '
                                 'OPTIONAL MATCH (n)-[r]-() '
                                 'RETURN n.id AS id, labels(n) AS labels, count(DISTINCT r) AS relationships',
                                 course=course_id, id=node_id, name=name))
    return {'checked_at': datetime.now(timezone.utc).isoformat(), 'database': database,
            'course_id': course_id, 'course_name': row['name'], 'revision': row['revision'],
            'selector': {'node_id': node_id, 'name': name}, 'reviewed_graph': reviewed,
            'native_graph': {'matching_nodes': len(native), 'nodes': [dict(n) for n in native]}}


def main():
    parser = argparse.ArgumentParser(description='只读核对审核图谱与 Neo4j 原始节点。不会删除任何数据。')
    parser.add_argument('--env', type=Path, default=DEFAULT_PROJECT / 'config/local.env', help='本地连接配置；兼容 key(2).env')
    parser.add_argument('--course-id', required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--node-id', help='推荐：节点资料表中的完整节点编号')
    selector.add_argument('--name', help='精确名称；同名节点可能不唯一，优先使用编号')
    parser.add_argument('--watch', type=float, metavar='SECONDS', help='持续查询间隔，至少 2 秒；Ctrl+C 结束')
    parser.add_argument('--before', type=Path, help='上次保存的 JSON 检查结果，用于确认删除前确实存在')
    parser.add_argument('--out', type=Path, help='保存本次 JSON 结果；不包含密码')
    args = parser.parse_args()
    if args.watch is not None and args.watch < 2:
        parser.error('--watch must be at least 2 seconds')
    values = read_env(args.env)
    uri = values['NEO4J_URI']; user = values.get('NEO4J_USER') or values.get('NEO4J_USERNAME')
    database = values.get('NEO4J_DATABASE') or 'neo4j'
    if not user or not values.get('NEO4J_PASSWORD'):
        parser.error('Missing Neo4j user/password in configuration file')
    from neo4j import GraphDatabase
    # Neo4j +s enables encrypted connections with certificate/hostname verification.
    parsed = urlsplit(uri)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error('URI must not contain credentials or parameters')
    if not (parsed.scheme in {'neo4j+s', 'bolt+s'} or
            parsed.scheme in {'neo4j', 'bolt'} and parsed.hostname in {'localhost', '127.0.0.1', '::1'}):
        parser.error('Remote connections must use verified TLS (+s)')
    options = {'connection_timeout': 8, 'connection_acquisition_timeout': 15, 'max_transaction_retry_time': 0}
    if uri.startswith(('neo4j+s://', 'bolt+s://')):
        import certifi
        from neo4j import TrustCustomCAs
        uri = uri.replace('+s://', '://', 1)
        options.update(encrypted=True, trusted_certificates=TrustCustomCAs(values.get('NEO4J_CA_BUNDLE') or certifi.where()))
    before = json.loads(args.before.read_text(encoding='utf-8')) if args.before else None
    if before and (before.get('course_id') != args.course_id or before.get('selector') != {'node_id': args.node_id, 'name': args.name}):
        parser.error('--before must refer to the same course and node selector')
    with GraphDatabase.driver(uri, auth=(user, values['NEO4J_PASSWORD']), **options) as driver:
        while True:
            report = query_database(driver, database, args.course_id, args.node_id, args.name)
            count = report['reviewed_graph']['matching_nodes']; native = report['native_graph']['matching_nodes']
            report['conclusion'] = ('审核图谱仍有该节点。' if count else
                                    '审核图谱未找到该节点；原始 Neo4j 节点仍存在，属于审核快照删除。' if native else
                                    '审核图谱和原始 Neo4j 都未找到该节点；仅凭当前结果不能证明它曾存在。')
            if before:
                report['deletion_confirmed_in_reviewed_graph'] = before['reviewed_graph']['matching_nodes'] > 0 and count == 0 and report['revision'] > before['revision']
                if report['deletion_confirmed_in_reviewed_graph']:
                    report['conclusion'] = '已对照删除前记录：审核图谱中的节点已删除。' + ('原始 Neo4j 节点仍保留。' if native else '原始 Neo4j 中也未找到该节点。')
            if count == 0 and report['reviewed_graph']['references_to_requested_id']:
                report['conclusion'] += '注意：审核图谱仍有关系引用该节点编号，需要清理。'
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            if not args.watch:
                return
            time.sleep(args.watch)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception as error:
        # Provider errors can contain sensitive connection details; expose only type/code.
        print(json.dumps({'success': False, 'error_type': type(error).__name__, 'error_code': getattr(error, 'code', None),
                          'note': '查询失败不能当作节点已删除；请检查网络、配置、课程与快照。'}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
