import json
import unittest
from unittest.mock import Mock
from backend.services.native_graph import apply_delta, read_graph, encode_snapshot, decode_snapshot, _read_pages
from neo4j.exceptions import ServiceUnavailable
from backend.routes.graph_contract import GraphError


class NativeGraphTests(unittest.TestCase):
    def graph(self):
        return {'nodes': [{'id':'a','name':'甲','type':'concept','description':'旧'},
                          {'id':'b','name':'乙','type':'concept'}],
                'links':[{'source':'a','target':'b','relation':'RELATED_TO'}]}

    def tx(self):
        tx=Mock();tx.run.return_value.single.return_value={'count':1};return tx

    def test_snapshot_codec_reads_legacy_and_compressed(self):
        g=self.graph()
        self.assertEqual(decode_snapshot(json.dumps(g)),g)
        self.assertEqual(decode_snapshot(encode_snapshot(g)),g)

    def test_native_node_update_is_course_scoped_and_parameterized(self):
        tx=self.tx();g=self.graph();changed=self.graph();changed['nodes'][0]['description']='修订'
        apply_delta(tx,g,changed,'course-a')
        query=tx.run.call_args.args[0];params=tx.run.call_args.kwargs
        self.assertIn('course_id:$course',query)
        self.assertEqual(params['course'],'course-a')
        self.assertEqual(params['rows'][0]['props']['description'],'修订')

    def test_relation_type_change_physically_replaces_relationship(self):
        tx=self.tx();g=self.graph();changed=self.graph();changed['links'][0]['relation']='PART_OF'
        apply_delta(tx,g,changed,'c')
        queries=[x.args[0] for x in tx.run.call_args_list]
        self.assertIn('DELETE r',queries[0]);self.assertIn('`RELATED_TO`',queries[0])
        self.assertTrue(any('MERGE (a)-[r:`PART_OF`]->(b)' in q for q in queries))

    def test_node_delete_removes_incident_edges(self):
        tx=self.tx();g=self.graph();changed={'nodes':[g['nodes'][1]],'links':[]}
        apply_delta(tx,g,changed,'c')
        self.assertTrue(any('DETACH DELETE n' in c.args[0] for c in tx.run.call_args_list))
        call=next(c for c in tx.run.call_args_list if 'DETACH DELETE' in c.args[0])
        self.assertEqual(call.kwargs['ids'],['a']);self.assertEqual(call.kwargs['course'],'c')

    def test_cross_course_node_rejected(self):
        g=self.graph();g['nodes'][0]['properties']={'course_id':'other'}
        with self.assertRaises(GraphError):apply_delta(self.tx(),{'nodes':[],'links':[]},g,'c')

    def test_bad_relationship_type_rejected(self):
        g=self.graph();g['links'][0]['relation']='X` DELETE n //'
        with self.assertRaises(GraphError):apply_delta(self.tx(),self.graph(),g,'c')

    def test_native_delete_is_not_revived_from_snapshot(self):
        session=Mock();session.run.side_effect=[
            [{'labels':['Concept'],'props':{'id':'b','course_id':'c','name':'数据库新名称','description':'新内容'}}], []]
        result=read_graph(session,self.graph(),'c')
        self.assertEqual([n['id'] for n in result['nodes']],['b'])
        self.assertEqual(result['nodes'][0]['name'],'数据库新名称')
        self.assertEqual(result['links'],[])

    def test_missing_relationship_endpoint_rejected(self):
        g=self.graph();g['links'][0]['target']='missing'
        with self.assertRaises(GraphError):apply_delta(self.tx(),self.graph(),g,'c')

    def test_failed_partial_page_does_not_duplicate_or_lose_rows(self):
        def broken():
            yield {'cursor':'b'}
            raise ServiceUnavailable('interrupted read')
        session=Mock();session.run.side_effect=[broken(),[{'cursor':'b'},{'cursor':'d'}],[{'cursor':'f'}]]
        from unittest.mock import patch
        with patch('backend.services.native_graph.time.sleep'):
            rows=list(_read_pages(session,'read','c',2))
        self.assertEqual([r['cursor'] for r in rows],['b','d','f'])
        self.assertEqual(session.run.call_args.kwargs['after'],'d')

    def test_read_retry_is_bounded(self):
        session=Mock();session.run.side_effect=ServiceUnavailable('offline')
        from unittest.mock import patch
        with patch('backend.services.native_graph.time.sleep'),self.assertRaises(ServiceUnavailable):
            list(_read_pages(session,'read','c'))
        self.assertEqual(session.run.call_count,3)

    def test_republication_uses_durable_review_not_already_overwritten_import(self):
        from backend.services.graph_repository import GraphRepository
        from unittest.mock import patch
        original=self.graph();review=self.graph();review['nodes'][0]['description']='教师修订'
        incoming=self.graph();incoming['nodes'][0]['description']='新模型说明'
        repo=GraphRepository({'KG_GRAPH_PATH':'unused','APP_DATA_DIR':'unused'})
        record={'model':original,'graph':incoming,'stored_graph':review,'revision':7}
        course={'id':'c','name':'课程'}
        with patch.object(repo,'resolve',return_value=course),patch.object(repo,'_records',return_value={'c':record}),patch.object(repo,'_save') as saved:
            saved.side_effect=lambda course,graph,*a,**kw:dict(graph,review_revision=8)
            repo.publish({'course_id':'c','course_name':'课程'},{'graph':incoming})
        self.assertEqual(saved.call_args.args[1]['nodes'][0]['description'],'教师修订')


if __name__=='__main__':unittest.main()
