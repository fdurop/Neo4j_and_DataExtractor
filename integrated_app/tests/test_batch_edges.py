import io,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
import fitz
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engines/extraction'))
from validate_full_document_api import _save_embedded
from backend.services import vector_service

class PDFMaskTests(unittest.TestCase):
    def check_image(self,alpha_base=False,smaller_mask=False):
        image=Image.new('RGBA',(40,30),(255,0,0,128));buffer=io.BytesIO();image.save(buffer,format='PNG')
        document=fitz.open();page=document.new_page();page.insert_image(fitz.Rect(0,0,40,30),stream=buffer.getvalue());info=page.get_images(full=True)[0]
        constructor=fitz.Pixmap
        def pixmap(*args,**kwargs):
            if len(args)==2 and args[0] is document:
                if args[1]==info[0] and alpha_base:return constructor(constructor(*args),1)
                if args[1]==info[1] and smaller_mask:return constructor(fitz.csGRAY,20,15,bytes([128])*300,False)
            return constructor(*args,**kwargs)
        with tempfile.TemporaryDirectory() as d,patch.object(fitz,'Pixmap',side_effect=pixmap):
            path=Path(d)/'asset.png';_save_embedded(document,info,path)
            with Image.open(path) as result:
                pixel=result.convert('RGBA').getpixel((10,10))
                self.assertEqual(result.size,(40,30));self.assertGreater(pixel[0],250);self.assertLessEqual(abs(pixel[3]-128),1)
        document.close()
    def test_already_alpha_base_keeps_soft_mask(self):self.check_image(alpha_base=True)
    def test_different_mask_resolution_keeps_coverage(self):self.check_image(smaller_mask=True)

class UploadScopeTests(unittest.TestCase):
    def test_search_is_restricted_to_requested_upload(self):
        settings={'EMBEDDING_MODEL':'model'}
        with patch.object(vector_service,'sync_graph'),patch.object(vector_service,'_connection',return_value=('https://index.pinecone.io',{})),patch.object(vector_service,'embed',return_value=[[1.0]]),patch.object(vector_service,'_post',return_value={'matches':[]}) as post:
            vector_service.search(settings,{'nodes':[]},'course','question',job_id='job-original')
            self.assertIn({'job_id':{'$eq':'job-original'}},post.call_args.args[2]['filter']['$and'])


class MetadataSyncTests(unittest.TestCase):
    def test_source_changes_are_synced_even_when_text_is_unchanged(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        stored = {}
        def post(url, headers, body):
            if url.endswith('/vectors/upsert'):
                stored.update({row['id']: row for row in body['vectors']})
            return {}
        transport = MagicMock()
        transport.get.side_effect = lambda *a, **kw: SimpleNamespace(status_code=200, json=lambda: {'vectors': stored})
        graph = {'nodes': [{'id': 'n', 'type': 'concept', 'name': '术语', 'description': '相同说明', 'properties': {'course_id': 'c', 'job_id': 'old'}}]}
        with tempfile.TemporaryDirectory() as directory:
            settings = {'APP_DATA_DIR': directory, 'EMBEDDING_MODEL': 'model', 'EMBEDDING_DIMENSION': 2, 'PINECONE_INDEX_NAME': 'index'}
            with patch.object(vector_service, '_connection', return_value=('https://index.pinecone.io', {})), patch.object(vector_service, 'embed', return_value=[[1.0, 0.0]]), patch.object(vector_service, '_post', side_effect=post), patch.object(vector_service, '_session', return_value=transport):
                vector_service.sync_graph(settings, graph, 'c')
                graph['nodes'][0]['properties']['job_id'] = 'new'
                result = vector_service.sync_graph(settings, graph, 'c')
                self.assertEqual(result['changed_count'], 1)
                self.assertEqual(next(iter(stored.values()))['metadata']['job_id'], 'new')

class SnapshotEnvelopeTests(unittest.TestCase):
    def test_archive_envelopes_do_not_reenter_domain_graph(self):
        from engines.graph.neo4j_runtime import clean_snapshot_graph, COURSE_SNAPSHOT_FIELDS
        import json
        graph={'nodes':[{'id':'c','type':'course','properties':{'name':'课程','review_graph_json':'x'*10000,'model_graph_json':'x'*10000,'previous_review_graph_json':'x'*10000}},
                        {'id':'n','type':'concept','description':'教师修正','properties':{'source':'p3'}}],
               'links':[{'source':'c','target':'n','relation':'CONTAINS'}]}
        clean=clean_snapshot_graph(graph)
        self.assertFalse(COURSE_SNAPSHOT_FIELDS.intersection(clean['nodes'][0]['properties']))
        self.assertEqual(clean['nodes'][1],graph['nodes'][1]);self.assertEqual(clean['links'],graph['links'])
        self.assertIn('previous_review_graph_json',graph['nodes'][0]['properties'])
        baseline=len(json.dumps(clean))
        for _ in range(20):
            clean['nodes'][0]['properties']['previous_review_graph_json']=json.dumps(clean)
            clean=clean_snapshot_graph(clean)
            self.assertEqual(len(json.dumps(clean)),baseline)

if __name__=='__main__':unittest.main()
