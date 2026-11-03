import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import requests
from backend.services import vector_service as service


class VectorRecoveryTests(unittest.TestCase):
    def test_transient_timeout_retries_only_three_times(self):
        session=Mock();session.post.side_effect=[requests.ReadTimeout(),requests.ReadTimeout(),Mock(status_code=200,json=lambda:{'ok':True})]
        with patch.object(service,'_session',return_value=session),patch.object(service.time,'sleep'):
            self.assertEqual(service._post('https://example',{},{}),{'ok':True})
        self.assertEqual(session.post.call_count,3)
        session.reset_mock();session.post.side_effect=requests.ReadTimeout()
        with patch.object(service,'_session',return_value=session),patch.object(service.time,'sleep'),self.assertRaises(RuntimeError):
            service._post('https://example',{}, {})
        self.assertEqual(session.post.call_count,3)

    def test_auth_failure_does_not_retry(self):
        session=Mock();session.post.return_value=Mock(status_code=401)
        with patch.object(service,'_session',return_value=session),self.assertRaises(RuntimeError):service._post('https://example',{}, {})
        self.assertEqual(session.post.call_count,1)

    def test_completed_batches_and_embedding_response_survive_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            settings={'APP_DATA_DIR':folder,'EMBEDDING_MODEL':'model','EMBEDDING_DIMENSION':2,'PINECONE_INDEX_NAME':'index'}
            graph={'nodes':[{'id':str(i),'name':'n'+str(i),'type':'concept','description':'text'+str(i)} for i in range(9)]}
            stored={};fail=[True];calls=[]
            def post(url,headers,payload):
                calls.append(payload)
                if url.endswith('/vectors/upsert'):
                    if len(payload['vectors'])==1 and fail[0]:fail[0]=False;raise RuntimeError('offline')
                    stored.update({r['id']:r for r in payload['vectors']})
                return {}
            embed=Mock(side_effect=lambda settings,texts,kind:[[1.0,0.0] for _ in texts])
            session=Mock();session.get.side_effect=lambda *args,**kw:SimpleNamespace(status_code=200,json=lambda:{'vectors':stored})
            with patch.object(service,'_connection',return_value=('https://example',{})),patch.object(service,'_post',side_effect=post),patch.object(service,'embed',embed),patch.object(service,'_session',return_value=session):
                with self.assertRaises(RuntimeError):service.sync_graph(settings,graph,'a')
                self.assertEqual(len(stored),8);self.assertEqual(embed.call_count,2)
                result=service.sync_graph(settings,graph,'a')
                self.assertEqual(embed.call_count,2) # cached response, no duplicate embedding request
                self.assertEqual(len(stored),9);self.assertEqual(result['changed_count'],1)
                self.assertEqual([len(c['vectors']) for c in calls],[8,1,1])
                self.assertEqual(service.sync_graph(settings,graph,'a')['changed_count'],0)
            self.assertFalse(list(Path(folder).rglob('pending_sync.json')))

    def test_pending_deleted_nodes_are_removed_when_graph_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'vector_indexes'/service._hash('a');root.mkdir(parents=True)
            settings={'APP_DATA_DIR':folder,'EMBEDDING_MODEL':'model','EMBEDDING_DIMENSION':2,'PINECONE_INDEX_NAME':'index'}
            old_id='course_'+service._hash('a:old')
            service._write_checkpoint(root/'pending_sync.json',{'model':'model','dimension':2,'index':'index','namespace':'','records':{old_id:'old'}})
            with patch.object(service,'_connection',return_value=('https://example',{})),patch.object(service,'_post',return_value={}) as post:
                service.sync_graph(settings,{'nodes':[]},'a')
            self.assertEqual(post.call_args.args[2]['ids'],[old_id])


if __name__=='__main__':unittest.main()
