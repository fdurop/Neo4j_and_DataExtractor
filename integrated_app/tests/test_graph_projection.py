import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

from backend.services.graph_projection import project_stored,stored_details,read_map
from backend.services.graph_repository import GraphRepository
from backend.routes.graph_contract import GraphError
from backend.app import create_app
from backend.models.db import db


class ProjectionTests(unittest.TestCase):
    def graph(self):
        return {'review_revision':4,'nodes':[
            {'id':'c','name':'概念','type':'concept','description':'可编辑说明','properties':{'course_id':'a','model_candidate_json':'large','importance_level':1,'source_page_ids':['p'],'evidence_node_ids':['m']}},
            {'id':'p','type':'page','name':'第1页','properties':{'raw_text':'应按需加载的原文','page_no':1,'job_id':'job_x','document_id':'d'}},
            {'id':'m','type':'figure','name':'图','properties':{'description':'视觉描述','source_page_id':'p'}}],
            'links':[{'source':'c','target':'m','relation':'HAS_EVIDENCE','properties':{'model_candidate_json':'large','description':'依据'}}]}

    def test_projection_keeps_topology_editable_text_roles_and_provenance(self):
        g=self.graph();small=project_stored(g)
        self.assertEqual([n['id'] for n in small['nodes']],['c','p','m'])
        self.assertEqual(small['links'][0]['relation'],'HAS_EVIDENCE')
        self.assertEqual(small['nodes'][0]['properties']['description'],'可编辑说明')
        self.assertEqual(small['nodes'][0]['properties']['importance_level'],1)
        self.assertEqual(small['nodes'][1]['properties']['page_no'],1)
        self.assertNotIn('raw_text',small['nodes'][1]['properties'])
        self.assertNotIn('model_candidate_json',small['nodes'][0]['properties'])
        self.assertEqual(g['nodes'][1]['properties']['raw_text'],'应按需加载的原文')

    def test_details_restore_source_text_and_material_evidence(self):
        details=stored_details(self.graph(),'c')
        self.assertEqual({n['id'] for n in details['nodes']},{'c','m','p'})
        page=next(n for n in details['nodes'] if n['id']=='p')
        self.assertEqual(page['properties']['raw_text'],'应按需加载的原文')
        self.assertEqual(len(details['links']),1)
        with self.assertRaises(GraphError):stored_details(self.graph(),'missing')

    def test_live_projection_queries_native_entities_and_stays_course_scoped(self):
        session=Mock();session.run.side_effect=[
            [{'labels':['Concept'],'props':{'id':'c','name':'实时名称','description':'实时说明','_app_type':'concept'}}],
            [{'source':'c','target':'p','kind':'HAS_EVIDENCE','props':{}}]]
        nodes,links=read_map(session,'a')
        self.assertEqual(nodes[0]['name'],'实时名称')
        self.assertEqual(nodes[0]['description'],'实时说明')
        self.assertEqual(links[0]['relation'],'HAS_EVIDENCE')
        self.assertEqual(session.run.call_count,2)
        for call in session.run.call_args_list:self.assertEqual(call.kwargs['course'],'a')
        self.assertNotIn('properties(n)',session.run.call_args_list[0].args[0])

    def test_pool_lifetime_is_repository_scoped(self):
        repo=GraphRepository({'APP_DATA_DIR':'unused','KG_GRAPH_PATH':'unused','NEO4J_URI':'x','NEO4J_USER':'u','NEO4J_PASSWORD':'p'})
        driver=Mock()
        with patch('backend.services.graph_repository.driver_for',return_value=driver) as factory:
            a=repo._driver();b=repo._driver();a.close();b.close()
            factory.assert_called_once();driver.close.assert_not_called()
            repo.close();driver.close.assert_called_once()

    def test_metadata_cache_requires_matching_live_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            repo=GraphRepository({'APP_DATA_DIR':directory,'KG_GRAPH_PATH':str(Path(directory)/'graphs.json')})
            repo.records_path.write_text(json.dumps({'a':{'revision':4,'graph':dict(self.graph(),integration={'name':'old'})}}),encoding='utf-8')
            session=Mock()
            old=repo._map_metadata(session,'a',4);self.assertEqual(old['integration']['name'],'old')
            session.run.assert_not_called()
            fresh=dict(self.graph(),review_revision=5,integration={'name':'fresh'})
            session.run.return_value.single.return_value={'review':json.dumps(fresh),'revision':5}
            result=repo._map_metadata(session,'a',5)
            self.assertEqual(result['integration']['name'],'fresh')
            self.assertNotIn('nodes',result);self.assertNotIn('links',result)
            result['integration']['name']='caller changed'
            self.assertEqual(repo._map_metadata(session,'a',5)['integration']['name'],'fresh')

    def test_map_does_not_read_stale_cached_topology_or_editable_text(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'courses.json').write_text(json.dumps([{'id':'a','name':'A'}]))
            repo=GraphRepository({'APP_DATA_DIR':directory,'KG_GRAPH_PATH':str(folder/'graphs.json'),'DATABASE_MODE':'neo4j'})
            repo._map_metadata_cache['a']={'revision':4,'metadata':{'hierarchy_candidates':[]}}
            session=Mock();session.run.return_value.single.return_value={'review':None,'revision':4,'native_version':1}
            driver=Mock();driver.session.return_value.__enter__=Mock(return_value=session);driver.session.return_value.__exit__=Mock(return_value=False)
            live=([{'id':'live','name':'live DB text'}],[])
            with patch.object(repo,'_driver',return_value=driver),patch('backend.services.graph_projection.read_map',return_value=live) as read:
                first=repo.get_map(course_id='a');second=repo.get_map(course_id='a')
            self.assertEqual(first['nodes'][0]['name'],'live DB text')
            self.assertEqual(second['nodes'][0]['id'],'live')
            self.assertEqual(read.call_count,2,'Native entities must still be queried on every load')

    def test_map_read_does_not_wait_for_writer_lock(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'courses.json').write_text(json.dumps([{'id':'a','name':'A'}]))
            repo=GraphRepository({'APP_DATA_DIR':directory,'KG_GRAPH_PATH':str(folder/'graphs.json'),'DATABASE_MODE':'neo4j'})
            repo._map_metadata_cache['a']={'revision':4,'metadata':{}}
            session=Mock();session.run.return_value.single.return_value={'review':None,'revision':4,'native_version':1}
            driver=Mock();driver.session.return_value.__enter__=Mock(return_value=session);driver.session.return_value.__exit__=Mock(return_value=False)
            with patch.object(repo,'_driver',return_value=driver),patch('backend.services.graph_projection.read_map',return_value=([],[])),ThreadPoolExecutor(1) as executor:
                repo.lock.acquire()
                try:self.assertEqual(executor.submit(repo.get_map,course_id='a').result(timeout=1)['review_revision'],4)
                finally:repo.lock.release()

    def test_revision_change_during_map_read_retries_once(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'courses.json').write_text(json.dumps([{'id':'a','name':'A'}]))
            repo=GraphRepository({'APP_DATA_DIR':directory,'KG_GRAPH_PATH':str(folder/'graphs.json'),'DATABASE_MODE':'neo4j'})
            session=Mock();session.run.return_value.single.side_effect=[
                {'review':None,'revision':4,'native_version':1},{'revision':5},
                {'review':None,'revision':5,'native_version':1},{'revision':5}]
            driver=Mock();driver.session.return_value.__enter__=Mock(return_value=session);driver.session.return_value.__exit__=Mock(return_value=False)
            with patch.object(repo,'_driver',return_value=driver),patch.object(repo,'_map_metadata',return_value={}),patch('backend.services.graph_projection.read_map',return_value=([],[])) as read:
                result=repo.get_map(course_id='a')
            self.assertEqual(result['review_revision'],5);self.assertEqual(read.call_count,2)

    def test_map_route_compression_auth_and_full_route_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            app=create_app({'APP_DATA_DIR':directory,'KG_GRAPH_PATH':str(Path(directory)/'graphs.json'),
                            'TESTING':True,'REQUIRE_AUTH':False})
            repo=Mock();repo.get_map.return_value=project_stored(self.graph());repo.get.return_value=self.graph()
            repo.get_details.return_value={'nodes':[],'links':[],'review_revision':4}
            app.extensions['graph_repository']=repo
            try:
                client=app.test_client();response=client.get('/api/graph?course_id=a&view=map',headers={'Accept-Encoding':'gzip'})
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.headers['Content-Encoding'],'gzip')
                self.assertEqual(json.loads(gzip.decompress(response.data))['projection'],'map-v1')
                self.assertEqual(response.headers['Cache-Control'],'no-store')
                self.assertIn('graph;dur=',response.headers['Server-Timing'])
                response=client.get('/api/graph?course_id=a');self.assertIn('model_candidate_json',response.json['nodes'][0]['properties'])
                response=client.get('/api/graph/node/c/details?course_id=a&revision=4');self.assertEqual(response.status_code,200)
                self.assertEqual(repo.get_details.call_args.kwargs['expected_revision'],4)
                response=client.get('/api/graph/node/c/details?course_id=a&revision=bad');self.assertEqual(response.status_code,400)
                app.config['REQUIRE_AUTH']=True
                self.assertEqual(client.get('/api/graph?course_id=a&view=map').status_code,401)
                self.assertEqual(client.get('/api/graph/node/c/details?course_id=a').status_code,401)
            finally:
                app.extensions['job_service'].shutdown()
                with app.app_context():db.session.remove();db.engine.dispose()


if __name__=='__main__':unittest.main()
