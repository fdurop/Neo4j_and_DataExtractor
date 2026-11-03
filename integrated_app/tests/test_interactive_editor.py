import json
from pathlib import Path
import sys
import tempfile
import unittest

APP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP))
from backend.services.graph_repository import GraphRepository
from backend.routes import graph_contract

class EditorPersistence(unittest.TestCase):
    def test_crud_stale_write_and_persistent_undo(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'courses.json').write_text(json.dumps([{'id':'a','name':'A'}]))
            config={'APP_DATA_DIR':directory,'KG_GRAPH_PATH':str(root/'graphs.json'),'DATABASE_MODE':'local'}
            repo=GraphRepository(config)
            repo.publish({'course_id':'a','course_name':'A'},{'graph':{'nodes':[],'links':[]}})
            for name in ['one','two']:
                repo.mutate(graph_contract.add_node,{'course_name':'A','name':name,'type':'concept'},expected_revision=repo.get(course_id='a')['review_revision'])
            graph=repo.get(course_id='a');one,two=[n['id'] for n in graph['nodes']]
            with self.assertRaises(graph_contract.GraphError):
                repo.mutate(graph_contract.edit_node,{'course_name':'A','id':one,'name':'stale'},expected_revision=1)
            repo.mutate(graph_contract.add_link,{'course_name':'A','source_id':one,'target_id':two,'relation':'RELATED_TO'})
            repo.mutate(graph_contract.delete_node,one,'A')
            self.assertEqual(repo.get(course_id='a')['links'],[])
            self.assertEqual(len(repo.get(course_id='a')['nodes']),1)
            restored=GraphRepository(config)
            restored.undo({'id':'a','name':'A'},restored.get(course_id='a')['review_revision'])
            self.assertEqual(len(restored.get(course_id='a')['nodes']),2)
            self.assertEqual(len(restored.get(course_id='a')['links']),1)
            # Continue undoing the relationship and both node creations.
            restored.undo({'id':'a','name':'A'},restored.get(course_id='a')['review_revision'])
            self.assertEqual(restored.get(course_id='a')['links'],[])
            self.assertEqual(len(restored.get(course_id='a')['nodes']),2)
            for expected in [1,0]:
                restored.undo({'id':'a','name':'A'},restored.get(course_id='a')['review_revision'])
                self.assertEqual(len(restored.get(course_id='a')['nodes']),expected)
            with self.assertRaises(graph_contract.GraphError):
                restored.undo({'id':'a','name':'A'},restored.get(course_id='a')['review_revision'])

if __name__=='__main__': unittest.main()
