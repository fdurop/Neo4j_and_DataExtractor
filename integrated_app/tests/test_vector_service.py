import unittest
from unittest.mock import patch
from backend.services import vector_service as service

class VectorServiceTests(unittest.TestCase):
    def graph(self):
        return {'nodes': [{'id': 'n1', 'type': 'text_chunk', 'name': '测试',
            'description': '教师修正后的知识', 'properties': {'course_id': 'c1',
            'raw_text': '已过期内容', 'origin': 'teacher_edit'}}]}

    def test_teacher_correction_replaces_old_evidence(self):
        record = service.records_for(self.graph(), 'c1')[0]
        self.assertIn('教师修正后的知识', record['text'])
        self.assertNotIn('已过期内容', record['text'])

    def test_cross_course_material_rejected(self):
        with self.assertRaises(ValueError):
            service.records_for(self.graph(), 'other')

    def test_removed_node_no_longer_has_vector_record(self):
        before = service.records_for(self.graph(), 'c1')
        after = service.records_for({'nodes': []}, 'c1')
        self.assertEqual(len(before), 1)
        self.assertEqual(after, [])

    def test_search_filters_course_and_embedding_model(self):
        settings = {'EMBEDDING_MODEL': 'model', 'PINECONE_NAMESPACE': ''}
        with patch.object(service, 'sync_graph'), patch.object(service, '_connection', return_value=('https://index.pinecone.io', {})), patch.object(service, 'embed', return_value=[[1.0]]), patch.object(service, '_post', return_value={'matches': [{'metadata': {'node_id': 'n1'}}]}) as post:
            self.assertEqual(service.search(settings, self.graph(), 'c1', 'question'), ['n1'])
            filters = post.call_args.args[2]['filter']['$and']
            self.assertIn({'course_id': {'$eq': 'c1'}}, filters)
            self.assertIn({'embedding_model': {'$eq': 'model'}}, filters)

if __name__ == '__main__': unittest.main()
