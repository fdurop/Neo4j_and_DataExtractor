import tempfile
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_graph_delete import inspect_snapshot, read_env


class DeleteVerificationTests(unittest.TestCase):
    def test_existing_node_and_incident_edges(self):
        graph = {'nodes': [{'id': 'a', 'name': 'A'}], 'links': [{'source': 'a', 'target': 'b'}]}
        self.assertEqual(inspect_snapshot(graph, node_id='a')['matching_nodes'], 1)
        self.assertEqual(inspect_snapshot(graph, node_id='a')['incident_relationships'], 1)

    def test_missing_node_does_not_hide_dangling_relationship(self):
        result = inspect_snapshot({'nodes': [], 'links': [{'source': {'id': 'gone'}, 'target': 'b'}]}, node_id='gone')
        self.assertEqual(result['matching_nodes'], 0)
        self.assertEqual(result['references_to_requested_id'], 1)

    def test_exact_name_does_not_match_substring(self):
        graph = {'nodes': [{'id': 'a', 'name': '磁极'}, {'id': 'b', 'name': '磁极位置'}]}
        self.assertEqual(inspect_snapshot(graph, name='磁极')['matching_node_ids'], ['a'])

    def test_nested_properties_and_ambiguous_names_are_counted(self):
        graph = {'nodes': [{'properties': {'id': 'a', 'canonical_name': 'A'}}, {'id': 'b', 'name': 'A'}]}
        self.assertEqual(inspect_snapshot(graph, name='A')['matching_nodes'], 2)

    def test_suffixed_credentials_and_quoted_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'key.env'
            path.write_text('# comment\nNEO4J_USER_2="user"\nNEO4J_PASSWORD_2=example\n', encoding='utf-8-sig')
            self.assertEqual(read_env(path), {'NEO4J_USER': 'user', 'NEO4J_PASSWORD': 'example'})


if __name__ == '__main__':
    unittest.main()
