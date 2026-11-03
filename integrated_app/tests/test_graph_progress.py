import unittest
from unittest.mock import Mock
from flask import Flask
from backend.services.graph_load_progress import LoadProgress
from backend.services.graph_projection import read_map
from backend.routes.knowledge_graph import kg_bp
from backend.routes.graph_contract import GraphError

class ProgressTests(unittest.TestCase):
    def test_progress_is_owned_expires_and_never_contains_graph_or_secret(self):
        store=LoadProgress();token='a'*32;update=store.start(token,1)
        update(phase='nodes',nodes=100,total_nodes=200,total_links=40)
        self.assertEqual(store.get(token,1)['nodes'],100)
        self.assertNotIn('owner',store.get(token,1))
        with self.assertRaises(GraphError):store.get(token,2)
        with self.assertRaises(GraphError):store.start('invalid',1)
        with self.assertRaises(GraphError):store.start(token,1)
        store.entries[token]['updated']-=181
        with self.assertRaises(GraphError):store.get(token,1)

    def test_counts_and_progress_follow_actual_rows(self):
        count=Mock();count.single.return_value={'nodes':1,'links':1}
        session=Mock();session.run.side_effect=[count,
            [{'labels':['Concept'],'props':{'id':'x','name':'X'}}],
            [{'source':'x','target':'x','kind':'RELATED_TO','props':{}}]]
        values=[];nodes,links=read_map(session,'course',lambda **v:values.append(v))
        self.assertEqual(len(nodes),1);self.assertEqual(len(links),1)
        self.assertEqual(values[1],dict(phase='nodes',nodes=0,links=0,total_nodes=1,total_links=1))
        self.assertEqual(values[-1],dict(phase='checking',nodes=1,links=1))
        self.assertEqual(session.run.call_count,3)
        for call in session.run.call_args_list:self.assertEqual(call.kwargs['course'],'course')

    def test_authenticated_progress_endpoint_and_unchanged_graph_payload(self):
        app=Flask(__name__);app.secret_key='test';app.config['REQUIRE_AUTH']=True
        app.register_blueprint(kg_bp,url_prefix='/api')
        repo=Mock();app.extensions['graph_repository']=repo
        graph={'nodes':[],'links':[]}
        def load(**kwargs):kwargs['progress'](phase='checking',nodes=0,links=0);return graph
        repo.get_map.side_effect=load;client=app.test_client();token='b'*32
        self.assertEqual(client.get('/api/graph/load-progress/'+token).status_code,401)
        with client.session_transaction() as s:s['user_id']=10
        response=client.get('/api/graph?view=map&progress_id='+token)
        self.assertEqual(response.status_code,200);self.assertEqual(response.json,graph)
        progress=client.get('/api/graph/load-progress/'+token)
        self.assertEqual(progress.json['phase'],'done');self.assertEqual(progress.headers['Cache-Control'],'no-store')
        with client.session_transaction() as s:s['user_id']=11
        self.assertEqual(client.get('/api/graph/load-progress/'+token).status_code,404)

    def test_failed_load_has_terminal_error_progress(self):
        app=Flask(__name__);app.secret_key='test';app.config['REQUIRE_AUTH']=False
        app.register_blueprint(kg_bp,url_prefix='/api');repo=Mock();app.extensions['graph_repository']=repo
        repo.get_map.side_effect=GraphError('数据库不可用',503);client=app.test_client();token='c'*32
        self.assertEqual(client.get('/api/graph?view=map&progress_id='+token).status_code,503)
        self.assertEqual(client.get('/api/graph/load-progress/'+token).json['phase'],'error')
