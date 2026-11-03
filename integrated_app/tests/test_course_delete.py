"""Destructive lifecycle regression tests: temporary data and mocked providers only."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, MagicMock, patch

from backend.app import create_app
from backend.models.db import db
from backend.services.graph_repository import GraphRepository
from backend.services import course_delete as deletion
from backend.services import qa
from flask import Flask
from types import SimpleNamespace


class DeleteCourseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.app = create_app({'TESTING': True, 'APP_DATA_DIR': str(self.root),
            'KG_GRAPH_PATH': str(self.root/'graphs.json'), 'AUTO_START_JOBS': False, 'DATABASE_MODE':'local'})
        self.store = self.app.extensions['course_store']
        self.repo = GraphRepository(self.app.config)
        self.app.extensions['graph_repository'] = self.repo
        self.a = self.store.add('A'); self.b = self.store.add('B')
        for c in (self.a, self.b):
            self.repo.publish({'course_id':c['id'], 'course_name':c['name']}, {'graph':{
                'nodes':[{'id':'shared', 'name':'共用名称', 'type':'concept', 'properties':{'course_id':c['id']}}], 'links':[]}})
        self.client = self.app.test_client()
        with self.app.app_context():
            from backend.models.user_model import User
            user = User(username='teacher', email='t@test.invalid', password_hash='unused', role='teacher')
            db.session.add(user); db.session.commit(); user_id = user.id
        with self.client.session_transaction() as session: session['user_id'] = user_id

    def tearDown(self):
        self.app.extensions['job_service'].shutdown()
        self.repo.close()
        with self.app.app_context(): db.session.remove(); db.engine.dispose()
        self.tmp.cleanup()

    def job(self, course):
        source=self.root/'incoming.pdf'; source.write_bytes(b'fake test source')
        service=self.app.extensions['job_service']
        job=service.create(course, 'source.pdf', source, 1)
        with service.lock:
            record=service._read(job['id']); record.update(status='succeeded',hidden=True); service._save(record)
        return service.get(job['id'], private=True)

    def delete(self): return self.client.delete('/api/courses/'+self.a['id'])

    def test_full_delete_includes_hidden_uploads_history_and_keeps_other_course(self):
        a_job=self.job(self.a); b_job=self.job(self.b)
        before=self.repo._records(self.b['id'])
        folder=self.root/'vector_indexes'/hashlib.sha256(self.a['id'].encode()).hexdigest()
        folder.mkdir(parents=True); (folder/'last_upsert.json').write_text('[]')
        result=self.delete(); self.assertEqual(result.status_code,200,result.json)
        self.assertTrue(result.json['deleted'])
        self.assertEqual(self.repo._records(self.a['id']),{})
        self.assertEqual(self.repo._records(self.b['id']),before)
        self.assertNotIn('A',json.loads((self.root/'graphs.json').read_text()))
        self.assertFalse(Path(a_job['job_dir']).exists()); self.assertTrue(Path(b_job['job_dir']).exists())
        self.assertFalse(folder.exists()); self.assertEqual([c['id'] for c in self.store.list()],[self.b['id']])
        self.assertEqual(self.client.get('/api/jobs/'+a_job['id']+'/source').status_code,404)
        self.assertEqual(self.client.post('/api/qa/',json={'course_id':self.a['id'],'question':'x'}).status_code,404)
        self.assertEqual(self.delete().status_code,404)

    def test_active_hidden_job_prevents_any_deletion(self):
        record=self.job(self.a); service=self.app.extensions['job_service']
        raw=service._read(record['id']); raw['status']='running'; service._save(raw)
        with patch.object(deletion,'_delete_vectors') as vectors:
            result=self.delete(); self.assertEqual(result.status_code,409); vectors.assert_not_called()
        self.assertFalse(self.store.resolve(self.a['id']).get('deleting'))

    def test_failure_keeps_retry_visible_and_blocks_use_even_after_restart(self):
        self.job(self.a)
        with patch.object(deletion,'_delete_vectors',side_effect=RuntimeError('offline')):
            self.assertEqual(self.delete().status_code,503)
        self.assertTrue(self.store.list()[0]['deleting'])
        self.assertEqual(self.client.post('/api/qa/',json={'course_id':self.a['id'],'question':'x'}).status_code,409)
        with patch.object(deletion,'_delete_vectors'):
            self.assertEqual(self.delete().status_code,200)

    def test_partial_folder_failure_retries_even_if_job_json_is_gone(self):
        job=self.job(self.a); original=deletion._remove_folder
        def partial(folder,root):
            (Path(job['job_dir'])/'job.json').unlink()
            raise OSError('file locked')
        with patch.object(deletion,'_remove_folder',side_effect=partial):
            self.assertEqual(self.delete().status_code,503)
        self.assertTrue(Path(job['job_dir']).exists())
        self.assertEqual(self.delete().status_code,200)
        self.assertFalse(Path(job['job_dir']).exists())

    def test_only_teacher_can_delete(self):
        self.assertEqual(self.app.test_client().delete('/api/courses/'+self.a['id']).status_code,401)
        with self.app.app_context():
            from backend.models.user_model import User
            User.query.filter_by(username='teacher').first().role='student'; db.session.commit()
        self.assertEqual(self.delete().status_code,403)

    def test_qa_request_and_delete_are_serialized(self):
        entered=threading.Event(); release=threading.Event(); deleting=threading.Event(); results=[]
        def runner(question,course): entered.set(); release.wait(4); return {'answer':'general'}
        self.app.config['QA_RUNNER']=runner
        user_id=None
        with self.client.session_transaction() as session: user_id=session['user_id']
        def ask():
            client=self.app.test_client()
            with client.session_transaction() as session: session['user_id']=user_id
            results.append(client.post('/api/qa/',json={'course_id':self.a['id'],'question':'x'}).status_code)
        def remove():
            client=self.app.test_client()
            with client.session_transaction() as session: session['user_id']=user_id
            deleting.set(); results.append(client.delete('/api/courses/'+self.a['id']).status_code)
        first=threading.Thread(target=ask); second=threading.Thread(target=remove)
        first.start(); self.assertTrue(entered.wait(2)); second.start(); deleting.wait(2)
        self.assertFalse(self.store.list()[0].get('deleting'))
        release.set(); first.join(5); second.join(5)
        self.assertFalse(first.is_alive()); self.assertFalse(second.is_alive()); self.assertEqual(results,[200,200])


class NativeAndVectorDeleteTests(unittest.TestCase):
    def test_native_deletion_is_transactional_and_checks_cross_course_edges(self):
        repo=MagicMock(); tx=repo._driver.return_value.session.return_value.__enter__.return_value.begin_transaction.return_value.__enter__.return_value
        tx.run.return_value.single.return_value={'count':3}
        from werkzeug.exceptions import Conflict
        with self.assertRaises(Conflict): deletion._native_delete(repo,'a')
        self.assertFalse(any('DELETE n' in c.args[0] for c in tx.run.call_args_list))
        tx.reset_mock(); tx.run.return_value.single.side_effect=[{'count':0},{'count':0},{'count':4}]
        self.assertEqual(deletion._native_delete(repo,'a'),4)
        tx.commit.assert_called_once()
        query=tx.run.call_args.args[0]
        self.assertIn('n.course_id=$id',query); self.assertIn('owner_course_id=$id',query)
        self.assertTrue(all(c.kwargs=={'id':'a'} for c in tx.run.call_args_list))

    def test_vectors_delete_only_course_filter_in_current_and_previous_destination(self):
        from backend.services import vector_service as vectors
        settings={'PINECONE_API_KEY':'fake','PINECONE_INDEX_NAME':'current','PINECONE_NAMESPACE':'now','EMBEDDING_DIMENSION':4}
        with patch.object(vectors,'_connection',return_value=('https://example',{})), patch.object(vectors,'_post',return_value={}) as post:
            deletion._delete_vectors(settings,'a',{'index':'old','namespace':'before','dimension':4})
        self.assertEqual(post.call_count,2)
        for call in post.call_args_list:
            self.assertEqual(call.args[2]['filter'],{'course_id':{'$eq':'a'}})
            self.assertNotIn('deleteAll',call.args[2])


class GeneralQaTests(unittest.TestCase):
    def run_answer(self,graph,response,vector_error=False):
        with tempfile.TemporaryDirectory() as folder:
            app=Flask(__name__); app.config.update(ENABLE_PAID_PIPELINE=True,DASHSCOPE_API_KEY='fake',APP_DATA_DIR=folder,
                API_MAX_COST_YUAN='3',API_MAX_REQUESTS=100,QWEN_MODEL='fake',ENABLE_VECTOR_RETRIEVAL=vector_error)
            app.extensions['graph_repository']=SimpleNamespace(get=lambda **kw:graph)
            with app.app_context(),patch.object(qa,'ValidationBudget'),patch.object(qa,'vision_config',return_value={}),patch.object(qa,'BudgetedQwenJSONClient') as client:
                client.return_value.chat_json.return_value=response
                if vector_error:
                    with patch('backend.services.vector_service.search',side_effect=RuntimeError('provider offline')):
                        with self.assertRaises(RuntimeError): qa.answer_question('绕组电阻',{'id':'a'})
                        client.assert_not_called()
                    return
                result=qa.answer_question('绕组电阻',{'id':'a'})
                client.return_value.chat_json.assert_called_once()
                return result

    def test_empty_course_calls_model_and_labels_general_without_sources(self):
        result=self.run_answer({'nodes':[]},{'answer':'一般原理','citations':[]})
        self.assertTrue(result['model_called']); self.assertEqual(result['answer_mode'],'general')
        self.assertEqual(result['sources'],[]); self.assertIn('通用知识',result['answer'])

    def test_unrelated_material_also_allows_general_answer(self):
        result=self.run_answer({'nodes':[{'id':'x','type':'concept','description':'完全无关'}]}, {'answer':'一般原理','citations':[]})
        self.assertEqual(result['answer_mode'],'general')

    def test_retrieval_failure_does_not_silently_answer_general(self):
        self.run_answer({'nodes':[{'id':'x'}]},None,vector_error=True)

    def test_general_cannot_fabricate_course_citations(self):
        with self.assertRaises(RuntimeError): self.run_answer({'nodes':[]},{'answer':'fake','citations':['S1']})


if __name__=='__main__': unittest.main()
