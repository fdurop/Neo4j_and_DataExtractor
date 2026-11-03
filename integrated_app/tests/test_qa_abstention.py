import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from flask import Flask
from backend.services import qa

class AbstentionTests(unittest.TestCase):
    def answer(self, response):
        with tempfile.TemporaryDirectory() as directory:
            app=Flask(__name__)
            app.config.update(ENABLE_PAID_PIPELINE=True,DASHSCOPE_API_KEY='unit-test-key',APP_DATA_DIR=directory,
                API_MAX_COST_YUAN='3',API_MAX_REQUESTS=100,QWEN_MODEL='qwen3.7-plus')
            app.extensions['graph_repository']=SimpleNamespace(get=lambda **kw:{'nodes':[
                {'id':'n','type':'text_chunk','description':'绕组电阻资料'}]})
            with app.app_context(),patch.object(qa,'ValidationBudget'),patch.object(qa,'vision_config',return_value={}),patch.object(qa,'BudgetedQwenJSONClient') as client:
                client.return_value.chat_json.side_effect=[response, {'answer':'通用知识说明；该课件具体电阻无法确认。','citations':[]}]
                return qa.answer_question('绕组电阻',{'id':'course'})

    def test_insufficient_course_evidence_uses_labeled_general_answer(self):
        result=self.answer({'answer':'猜测电阻是999欧姆','citations':[],'insufficient_evidence':True})
        self.assertTrue(result['insufficient_evidence'])
        self.assertNotIn('999',result['answer'])
        self.assertEqual(result['sources'],[])
        self.assertEqual(result['answer_mode'], 'general')
        self.assertIn('通用知识', result['answer'])

    def test_normal_claim_without_citation_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.answer({'answer':'电阻是999欧姆','citations':[],'insufficient_evidence':False})

    def test_invented_citation_is_rejected_even_for_abstention(self):
        with self.assertRaises(RuntimeError):
            self.answer({'answer':'资料不足','citations':['invented'],'insufficient_evidence':True})

    def test_grounded_answer_still_requires_actual_source(self):
        result=self.answer({'answer':'课件资料','citations':['S1'],'insufficient_evidence':False})
        self.assertEqual(result['sources'][0]['node_id'],'n')
        self.assertFalse(result['insufficient_evidence'])

if __name__=='__main__':unittest.main()
