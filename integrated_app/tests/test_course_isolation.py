"""Isolation regression tests: temporary stores only, no cloud credentials/network."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from backend.routes import graph_contract as gc
from backend.services.graph_repository import GraphRepository
from backend.services.native_graph import apply_delta
from backend.app import create_app
from backend.models.db import db


def graph(course):
    # Intentionally identical IDs/names in different courses.
    return {'nodes': [{'id': 'shared', 'name': 'PWM', 'type': 'concept',
                       'description': course, 'properties': {'course_id': course}},
                      {'id': 'peer', 'name': '电机', 'type': 'concept',
                       'properties': {'course_id': course}}],
            'links': [{'source': 'shared', 'target': 'peer', 'relation': 'RELATED_TO',
                       'relation_id': 'same-relation', 'description': course}]}


class CourseIsolation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root/'courses.json').write_text(json.dumps([{'id': 'a', 'name': 'A'}, {'id': 'b', 'name': 'B'}]))
        self.repo = GraphRepository({'APP_DATA_DIR': self.tmp.name, 'KG_GRAPH_PATH': str(root/'graphs.json'),
                                     'DATABASE_MODE': 'local'})
        for c in ('a', 'b'):
            self.repo.publish({'course_id': c, 'course_name': c.upper()}, {'graph': graph(c)})
        self.b = copy.deepcopy(self.repo._records('b'))

    def tearDown(self):
        self.repo.close()
        self.tmp.cleanup()

    def assert_b_unchanged(self):
        self.assertEqual(self.repo._records('b'), self.b)

    def test_crud_and_consecutive_undo_with_identical_ids(self):
        self.assertEqual(self.repo.get(course_id='a')['nodes'][0]['description'], 'a')
        self.assertEqual(self.repo.get(course_id='b')['nodes'][0]['description'], 'b')
        ops = [(gc.add_node, {'course_name': 'A', 'name': '新增'}),
               (gc.edit_node, {'course_name': 'A', 'id': 'shared', 'description': '修正'}),
               (gc.add_link, {'course_name': 'A', 'source_id': 'peer', 'target_id': 'shared', 'relation': 'PART_OF'}),
               (gc.edit_link, {'course_name': 'A', 'relation_id': 'same-relation', 'relation': 'USES'}),
               (gc.delete_link, {'course_name': 'A', 'source_id': 'shared', 'target_id': 'peer', 'old_relation': 'RELATED_TO'}),
               (gc.delete_node, 'shared', 'A')]
        for operation, *args in ops:
            before = self.repo.get(course_id='a')
            self.repo.mutate(operation, *args)
            self.assert_b_unchanged()
            # Reload the repository so undo must use persisted history.
            fresh = GraphRepository(self.repo.config)
            fresh.undo({'id': 'a', 'name': 'A'}, fresh.get(course_id='a')['review_revision'])
            after = fresh.get(course_id='a')
            for key in ('nodes', 'links'):
                self.assertEqual(after[key], before[key])
            self.assert_b_unchanged()
        self.repo.mutate(gc.delete_node, 'shared', 'A')
        self.repo.mutate(gc.edit_node, {'course_name': 'A', 'id': 'peer', 'description': 'second'})
        for _ in range(2):
            self.repo.undo({'id': 'a', 'name': 'A'}, self.repo.get(course_id='a')['review_revision'])
            self.assert_b_unchanged()
        self.assertEqual(len(self.repo.get(course_id='a')['links']), 1)

    def test_missing_course_does_not_discover_target_in_other_courses(self):
        for operation, args in [(gc.edit_node, ({'id': 'shared', 'description': 'bad'},)),
                                (gc.delete_node, ('shared',)),
                                (gc.delete_link, ({'relation_id': 'same-relation'},))]:
            with self.subTest(operation=operation.__name__), self.assertRaises(gc.GraphError):
                self.repo.mutate(operation, *args)
        self.assert_b_unchanged()

    def test_unknown_endpoint_and_forged_undo_course_rejected(self):
        with self.assertRaises(gc.GraphError):
            self.repo.mutate(gc.add_link, {'course_name': 'A', 'source_id': 'shared',
                                         'target_id': 'foreign-only', 'relation': 'USES'})
        with self.assertRaises(gc.GraphError):
            self.repo.undo({'id': 'b', 'name': 'A'}, 1)
        self.assert_b_unchanged()

    def test_poisoned_undo_snapshot_rejected_before_cache_write(self):
        self.repo.mutate(gc.delete_node, 'shared', 'A')
        records = json.loads(self.repo.records_path.read_text(encoding='utf-8'))
        for key in ('previous_graph',):
            records['a'][key]['nodes'][0]['properties']['course_id'] = 'b'
        records['a']['undo_stack'][-1]['nodes'][0]['properties']['course_id'] = 'b'
        self.repo.records_path.write_text(json.dumps(records), encoding='utf-8')
        before = self.repo.records_path.read_bytes()
        with self.assertRaises(gc.GraphError):
            self.repo.undo({'id': 'a', 'name': 'A'}, self.repo.get(course_id='a')['review_revision'])
        self.assertEqual(self.repo.records_path.read_bytes(), before)
        self.assert_b_unchanged()

    def test_api_rejects_conflicting_courses_for_reads_crud_and_undo(self):
        app = create_app({**self.repo.config, 'TESTING': True, 'REQUIRE_AUTH': False})
        app.extensions['graph_repository'] = self.repo
        client = app.test_client()
        try:
            cases = [('get', '/api/graph?course_id=a&course_name=B', None),
                     ('get', '/api/graph/node/shared/details?course_id=a&course_name=B', None),
                     ('post', '/api/graph/node?course_id=a', {'course_name': 'B', 'name': 'bad'}),
                     ('put', '/api/graph/node?course_id=a', {'course_name': 'B', 'id': 'shared', 'name': 'bad'}),
                     ('delete', '/api/graph/node/shared?course_id=a&course_name=B', None),
                     ('post', '/api/graph/link?course_id=a', {'course_name': 'B', 'source_id': 'shared', 'target_id': 'peer'}),
                     ('put', '/api/graph/link?course_id=a', {'course_name': 'B', 'relation_id': 'same-relation', 'description': 'bad'}),
                     ('delete', '/api/graph/link?course_id=a', {'course_name': 'B', 'relation_id': 'same-relation'}),
                     ('post', '/api/graph/undo?course_id=a', {'course_name': 'B'}),
                     ('post', '/api/graph/undo?course_id=a&course_name=B', {'course_name': 'A'}),
                     ('put', '/api/graph/node', {'id': 'shared', 'description': 'bad'})]
            for method, url, payload in cases:
                with self.subTest(method=method, url=url):
                    response = getattr(client, method)(url, json=payload, headers={'X-Graph-Revision': '1'})
                    self.assertIn(response.status_code, (400, 404, 409))
                    self.assert_b_unchanged()
            good = client.put('/api/graph/node?course_id=a', json={'id': 'shared', 'description': 'only A'})
            self.assertEqual(good.status_code, 200)
            self.assertEqual(client.get('/api/graph/node/shared/details?course_id=b').json['nodes'][0]['description'], 'b')
        finally:
            app.extensions['job_service'].shutdown()
            with app.app_context():
                db.session.remove(); db.engine.dispose()


class NativeIsolation(unittest.TestCase):
    def test_foreign_snapshot_rejected_before_any_cypher(self):
        for location in ('node', 'node_property', 'link', 'link_property', 'endpoint', 'root'):
            with self.subTest(location=location):
                bad = graph('a')
                if location == 'node': bad['nodes'][0]['course_id'] = 'b'
                if location == 'node_property': bad['nodes'][0]['properties']['course_id'] = 'b'
                if location == 'link': bad['links'][0]['course_id'] = 'b'
                if location == 'link_property': bad['links'][0]['properties'] = {'course_id': 'b'}
                if location == 'endpoint': bad['links'][0]['target'] = 'foreign'
                if location == 'root': bad['nodes'][0]['labels'] = ['Course']
                tx = Mock()
                with self.assertRaises(gc.GraphError): apply_delta(tx, graph('a'), bad, 'a', force=True)
                tx.run.assert_not_called()

    def test_detach_refuses_cross_course_or_unowned_relationship(self):
        tx = Mock();tx.run.return_value.single.return_value = {'outside': 1}
        with self.assertRaises(gc.GraphError):
            apply_delta(tx, graph('a'), {'nodes': [], 'links': []}, 'a')
        self.assertEqual(tx.run.call_count, 1)
        self.assertNotIn('DELETE', tx.run.call_args.args[0])
        self.assertEqual(tx.run.call_args.kwargs['course'], 'a')

    def test_all_native_delta_writes_scope_nodes_and_both_edge_endpoints(self):
        tx = Mock()
        tx.run.return_value.single.return_value = {'outside': 0, 'count': 1}
        before = graph('a'); after = copy.deepcopy(before)
        after['nodes'][0]['description'] = 'edit'
        after['links'][0]['relation'] = 'USES'
        apply_delta(tx, before, after, 'a')
        for call in tx.run.call_args_list:
            self.assertEqual(call.kwargs['course'], 'a')
            self.assertIn('course_id:$course', call.args[0])
            if 'DELETE r' in call.args[0] or 'MERGE (a)-' in call.args[0]:
                self.assertEqual(call.args[0].count('course_id:$course'), 2)


if __name__ == '__main__':
    unittest.main()
